"""
SpreadsheetBench API server - From GitHub RUCKBReasoning/SpreadsheetBench
Provides HTTP API for code execution with conversation-based kernel management.
"""

import os
import time
import json
import signal
import logging
import argparse
import tornado.ioloop
import tornado.web
import tornado.httpserver
from collections import namedtuple
from .jupyter import JupyterKernel, JupyterKernelWrapper

logger = logging.getLogger(__name__)

# Global data structure to map convid to (JupyterKernelWrapper, JupyterKernel)
JupyterKernelType = namedtuple(
    "JupyterKernelType", ["kernel_wrapper", "kernel", "last_access_time"]
)


def cleanup_kernels(app, force=False):
    """Cleanup kernels and gateway dockers that have timed out."""
    KERNEL_TIMEOUT = 10 * 60  # 10 minutes
    current_time = time.time()
    to_delete = []
    conv_id_to_kernel = app.conv_id_to_kernel

    # Find all kernels that have timed out
    for convid in conv_id_to_kernel.keys():
        last_access = conv_id_to_kernel[convid].last_access_time
        if current_time - last_access > KERNEL_TIMEOUT:
            to_delete.append(convid)

    if force:
        to_delete = list(conv_id_to_kernel.keys())
        logger.info(f"Force cleanup all {len(to_delete)} kernels")

    for convid in to_delete:
        # Close the JupyterKernelWrapper by closing its context manager
        kernel_wrapper = conv_id_to_kernel[convid].kernel_wrapper
        kernel_wrapper.__exit__(None, None, None)  # Close the JupyterKernelWrapper
        # Delete the entry from the global data structure
        del conv_id_to_kernel[convid]
        logger.info(f"Kernel closed for conversation {convid}")


class ExecuteHandler(tornado.web.RequestHandler):
    """Handle code execution requests in SpreadsheetBench format."""

    async def post(self):
        data = json.loads(self.request.body)
        convid = data.get("convid")
        code = data.get("code")

        if not convid or not code:
            self.set_status(400)
            self.write(json.dumps({"error": "Missing required fields: convid and code"}))
            return

        # Create a new kernel if not exist
        new_kernel = False

        conv_id_to_kernel = self.application.conv_id_to_kernel
        if convid not in conv_id_to_kernel:
            try:
                kernel_wrapper = JupyterKernelWrapper(
                    name=f"conv-{convid}",
                )
                url_suffix = kernel_wrapper.__enter__()
                if os.environ.get("DEBUG", False):
                    logger.info(f"Kernel URL: {url_suffix}")

                kernel = JupyterKernel(url_suffix, convid)
                await kernel.initialize()

                conv_id_to_kernel[convid] = JupyterKernelType(kernel_wrapper, kernel, None)
                new_kernel = True
                logger.info(f"Kernel created for conversation {convid}")
            except Exception as e:
                logger.error(f"Failed to create kernel for {convid}: {e}")
                self.set_status(500)
                self.write(json.dumps({"error": f"Failed to create kernel: {str(e)}"}))
                return

        # Update last access time
        kernel_access_time = time.time()
        conv_id_to_kernel[convid] = conv_id_to_kernel[convid]._replace(
            last_access_time=kernel_access_time
        )

        # Execute the code
        try:
            kernel: JupyterKernel = conv_id_to_kernel[convid].kernel
            result = await kernel.execute(code)

            self.write(json.dumps({"result": result, "new_kernel_created": new_kernel}))
        except Exception as e:
            logger.error(f"Code execution failed for {convid}: {e}")
            self.set_status(500)
            self.write(json.dumps({"error": f"Code execution failed: {str(e)}"}))


class HealthHandler(tornado.web.RequestHandler):
    """Health check endpoint."""

    def get(self):
        self.write(
            json.dumps(
                {
                    "status": "healthy",
                    "service": "spreadsheetbench-api",
                    "kernels_active": len(self.application.conv_id_to_kernel),
                }
            )
        )


class KernelsHandler(tornado.web.RequestHandler):
    """List active kernels."""

    def get(self):
        conv_id_to_kernel = self.application.conv_id_to_kernel
        kernel_info = {}

        for convid, kernel_data in conv_id_to_kernel.items():
            kernel_info[convid] = {
                "last_access": kernel_data.last_access_time,
                "age_seconds": time.time() - (kernel_data.last_access_time or time.time()),
            }

        self.write(json.dumps({"active_kernels": kernel_info, "total_count": len(kernel_info)}))


def create_spreadsheetbench_app():
    """Create the SpreadsheetBench Tornado application."""
    app = tornado.web.Application(
        [
            (r"/execute", ExecuteHandler),
            (r"/health", HealthHandler),
            (r"/kernels", KernelsHandler),
        ]
    )
    app.conv_id_to_kernel = {}

    # Setup periodic cleanup
    periodic_cleanup = tornado.ioloop.PeriodicCallback(
        lambda: cleanup_kernels(app), int(os.environ.get("CLEANUP_TIMEOUT_MS", 60000))
    )
    periodic_cleanup.start()

    # Setup signal handler only if in main thread
    import threading

    if threading.current_thread() is threading.main_thread():

        def signal_handler(signum, frame, app):
            logger.info("Received SIGINT, cleaning up...")
            cleanup_kernels(app, force=True)
            tornado.ioloop.IOLoop.current().stop()
            logger.info("Cleanup complete, shutting down.")

        signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, app))

    return app


def run_spreadsheetbench_server(port=8000):
    """Run the SpreadsheetBench API server standalone."""
    app = create_spreadsheetbench_app()

    server = tornado.httpserver.HTTPServer(app)
    server.listen(port)
    logger.info(f"SpreadsheetBench API server started on port {port}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_spreadsheetbench_server(args.port)
