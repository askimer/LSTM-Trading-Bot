# OpenCode CLI Installation Guide

## Installation Completed Successfully

The OpenCode CLI tool has been successfully installed on this system.

### Version Information
- OpenCode CLI Version: 0.0.55
- Installation Date: January 24, 2026
- Platform: macOS

### Installation Method
Installed using the official install script from the OpenCode repository:
```bash
curl -fsSL https://raw.githubusercontent.com/opencode-ai/opencode/refs/heads/main/install | bash
```

### Important Notes
- The OpenCode CLI executable is located in `/Users/askimer/.local/bin/opencode`
- The installation automatically added the path to `$PATH` in `/Users/askimer/.profile`
- To use the CLI in the current session, you need to source the profile: `source ~/.profile`
- For permanent availability, you may need to add the path to your shell profile (e.g., `.zshrc` or `.bashrc`)

### Available Commands
The OpenCode CLI provides the following main options:
- Interactive mode: `opencode` (runs the TUI interface)
- Non-interactive mode: `opencode -p "prompt"`
- Debug mode: `opencode -d`
- Version check: `opencode --version`
- Help: `opencode --help`

### Dependencies
The installation warns about missing optional dependencies:
- Ripgrep (rg): For enhanced search capabilities
- FZF: For fuzzy finder functionality

### Verification
Installation can be verified by running:
```bash
source ~/.profile && opencode --version
```

### Configuration
To use OpenCode with AI providers, you'll need to configure API keys for services like OpenAI, Anthropic, Google Gemini, etc. The tool supports multiple AI providers including:
- OpenAI
- Anthropic Claude
- Google Gemini
- AWS Bedrock
- Groq
- Azure OpenAI
- OpenRouter

### Features
OpenCode provides:
- Interactive TUI built with Bubble Tea
- Multiple AI provider support
- Session management
- Tool integration (command execution, file search/modification)
- LSP integration for code intelligence
- File change tracking
- External editor support