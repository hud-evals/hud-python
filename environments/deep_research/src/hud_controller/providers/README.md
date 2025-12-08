# Remote Browser Providers

This directory contains implementations for various cloud browser providers that can be used with the HUD Remote Browser environment.

## Supported Providers

### 1. **AnchorBrowser** ✅ (Implemented)
- **API Endpoint**: `https://api.anchorbrowser.io/v1/sessions`
- **Features**:
  - Residential proxy support
  - CAPTCHA solving
  - Ad blocking
  - Popup blocking
- **API Key**: `ANCHOR_API_KEY` environment variable
- **Documentation**: Internal

### 2. **BrowserBase** 🚧 (To be implemented)
- **API Endpoint**: `https://api.browserbase.com/v1/sessions`
- **Features**:
  - Multiple regions support
  - Context persistence
  - Live view URLs
  - Session recordings
  - Proxy support
- **API Key**: `X-BB-API-Key` header
- **Documentation**: https://docs.browserbase.com/reference/api/create-a-session

### 3. **HyperBrowser** 🚧 (To be implemented)
- **API Endpoint**: `https://api.hyperbrowser.ai/api/session`
- **Features**:
  - Stealth mode
  - Advanced proxy configuration (country/state/city)
  - Profile management
  - Web recording
  - CAPTCHA solving
  - Ad blocking
  - Browser fingerprinting
- **API Key**: `x-api-key` header
- **Documentation**: https://docs.hyperbrowser.ai/reference/api-reference/sessions

### 4. **Steel** 🚧 (To be implemented)
- **API Endpoint**: `https://api.steel.dev/v1/sessions`
- **Features**:
  - Session management
  - Browser automation
  - Proxy support
- **API Key**: `steel_api_key` header or `STEEL_API_KEY` env variable
- **Documentation**: https://docs.steel.dev/api-reference

### 5. **Kernel** ❌ (Not yet available)
- **Status**: API not yet available for browser sessions
- **Documentation**: N/A

## Provider Lifecycle

Each provider follows a similar lifecycle pattern:

1. **Initialization**
   - Set up API credentials
   - Configure base URLs and default options

2. **Session Creation** (`launch()`)
   - Make API request to create a new browser session
   - Handle provider-specific options (proxy, stealth, etc.)
   - Return CDP WebSocket URL for Playwright connection

3. **Session Management**
   - Track session IDs and metadata
   - Provide status checks
   - Handle session-specific features (live view, recordings, etc.)

4. **Session Termination** (`close()`)
   - Clean up resources
   - End the browser session via API
   - Handle any provider-specific cleanup

## Implementation Guide

To add a new provider:

1. Create a new file in this directory (e.g., `browserbase.py`)
2. Inherit from `BrowserProvider` base class
3. Implement required methods:
   - `__init__()` - Initialize with API credentials
   - `launch()` - Create a new session and return CDP URL
   - `close()` - Terminate the session
   - `get_status()` - Return session status
4. Add provider to the registry in `__init__.py`
5. Update environment variables in the main README

## Environment Variables

Each provider uses specific environment variables:

- **AnchorBrowser**: `ANCHOR_API_KEY`
- **BrowserBase**: `BROWSERBASE_API_KEY`
- **HyperBrowser**: `HYPERBROWSER_API_KEY`
- **Steel**: `STEEL_API_KEY`

## Common Features Across Providers

| Feature | AnchorBrowser | BrowserBase | HyperBrowser | Steel |
|---------|---------------|-------------|--------------|-------|
| Proxy Support | ✅ | ✅ | ✅ | ✅ |
| CAPTCHA Solving | ✅ | ❓ | ✅ | ❓ |
| Ad Blocking | ✅ | ❓ | ✅ | ❓ |
| Session Recording | ❌ | ✅ | ✅ | ❓ |
| Live View | ✅ | ✅ | ✅ | ❓ |
| Profile Persistence | ❌ | ✅ | ✅ | ❓ |
| Multi-Region | ❌ | ✅ | ✅ | ❓ |