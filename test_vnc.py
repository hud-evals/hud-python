#!/usr/bin/env python3
"""Test VNC setup in kernel provider"""

import asyncio
import sys
import os
sys.path.append('/app/src')

from hud_controller.providers.kernel import KernelProvider

async def test_vnc():
    provider = KernelProvider()

    try:
        print("Starting kernel provider...")
        cdp_url = await provider.launch()
        print(f"✅ Browser launched with CDP: {cdp_url}")
        print("✅ VNC should be available at: http://localhost:8080/vnc.html")

        # Keep running for testing
        print("Keeping alive for VNC testing... (Ctrl+C to stop)")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Shutting down...")
        provider.close()
    except Exception as e:
        print(f"❌ Error: {e}")
        provider.close()

if __name__ == "__main__":
    os.environ["DISPLAY"] = ":1"
    asyncio.run(test_vnc())