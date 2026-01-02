const { Plugin, ItemView, WorkspaceLeaf, Notice } = require('obsidian');

const VIEW_TYPE_CHAT = "g-retriever-chat-view";

class GRetrieverChatView extends ItemView {
    constructor(leaf, plugin) {
        super(leaf);
        this.plugin = plugin;
        this.chatHistory = [];
    }

    getViewType() {
        return VIEW_TYPE_CHAT;
    }

    getDisplayText() {
        return "G-Retriever Chat";
    }

    getIcon() {
        return "message-square";
    }

    async onOpen() {
        const container = this.containerEl.children[1];
        container.empty();
        container.addClass('g-retriever-chat-container');

        // Chat History
        const chatBox = container.createDiv({ cls: 'g-retriever-chat-box' });
        this.chatBox = chatBox;

        // Input Area
        const inputContainer = container.createDiv({ cls: 'g-retriever-input-container' });

        const input = inputContainer.createEl('textarea', {
            cls: 'g-retriever-input',
            attr: {
                placeholder: 'Ask a question about your notes...',
                rows: '3'
            }
        });
        this.input = input;

        const sendButton = inputContainer.createEl('button', {
            text: 'Send',
            cls: 'g-retriever-send-button'
        });

        sendButton.addEventListener('click', () => this.sendMessage());

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Check backend status
        this.checkBackendStatus();
    }

    async checkBackendStatus() {
        try {
            const url = await this.plugin.getBackendUrl();
            const response = await fetch(`${url}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(3000)
            });

            if (response.ok) {
                this.addMessage('system', '✓ G-Retriever connected and ready!');
            } else {
                this.addMessage('system', '⚠️ Backend responded but may not be ready');
            }
        } catch (error) {
            this.addMessage('system', '❌ Backend not responding. Please start server.py first!');
            new Notice('G-Retriever backend not found. Run: python server.py');
        }
    }

    async sendMessage() {
        const question = this.input.value.trim();
        if (!question) return;

        // User message
        this.addMessage('user', question);
        this.input.value = '';

        // Create debug section container
        const debugMsgId = this.addDebugSection();

        try {
            // Call Python backend with progress updates
            await this.queryBackendWithProgress(question, debugMsgId);

        } catch (error) {
            this.addToDebugSection(debugMsgId, `❌ Error: ${error.message}`, 'error');
            this.addMessage('assistant', `Error: ${error.message}`);
            new Notice('Error communicating with G-Retriever backend');
            console.error('G-Retriever error:', error);
        }
    }

    addDebugSection() {
        const debugId = `debug-${Date.now()}-${Math.floor(Math.random() * 1000000)}`;

        const debugDiv = this.chatBox.createDiv({
            cls: 'g-retriever-debug-section'
        });
        debugDiv.id = debugId;

        return debugId;
    }

    addToDebugSection(debugId, text, className = '') {
        const debugDiv = this.chatBox.querySelector(`#${debugId}`);
        if (debugDiv) {
            const line = debugDiv.createDiv({
                cls: `g-retriever-debug-line ${className}`
            });
            line.textContent = text;
            this.chatBox.scrollTop = this.chatBox.scrollHeight;
        }
    }

    async queryBackendWithProgress(question, debugMsgId) {
        const url = await this.plugin.getBackendUrl();

        // Show query
        this.addToDebugSection(debugMsgId, `\nQuery: ${question}`, 'query');

        // Show progress steps
        this.addToDebugSection(debugMsgId, 'Retrieving relevant nodes...', 'step');

        // Longer timeout for generation (2 minutes)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000);

        try {
            const response = await fetch(`${url}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || 'Backend error');
            }

            const result = await response.json();

            console.log('Received result:', result);
            console.log('Answer:', result.answer);
            console.log('Answer length:', result.answer ? result.answer.length : 'undefined');

            // Show retrieved node names
            if (result.debug_info && result.debug_info.retrieved_names) {
                this.addToDebugSection(debugMsgId,
                    `DEBUG: Retrieved node names: ${JSON.stringify(result.debug_info.retrieved_names)}`,
                    'debug');
            }

            this.addToDebugSection(debugMsgId, 'Constructing subgraph...', 'step');
            this.addToDebugSection(debugMsgId, 'Generating answer...', 'step');

            // Show the answer - make sure it exists
            if (result.answer) {
                this.addMessage('assistant', result.answer);
            } else {
                console.error('No answer in result!');
                this.addMessage('assistant', 'Error: No answer received from backend');
            }

            // Show used notes
            if (result.subgraph_nodes && result.subgraph_nodes.length > 0) {
                const sources = result.subgraph_nodes.slice(0, 5).join(', ');
                this.addToDebugSection(debugMsgId,
                    `\nVerwendete Notizen: ${sources}`,
                    'sources');
            }

        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new Error('Request timeout - generation took too long');
            }
            throw error;
        }
    }

    addMessage(role, content, className = '') {
        const messageId = `msg-${Date.now()}-${Math.floor(Math.random() * 1000000)}`;

        const messageDiv = this.chatBox.createDiv({
            cls: `g-retriever-message g-retriever-${role} ${className}`
        });
        messageDiv.id = messageId;

        const contentDiv = messageDiv.createDiv({ cls: 'g-retriever-message-content' });
        contentDiv.innerHTML = this.formatContent(content);

        this.chatBox.scrollTop = this.chatBox.scrollHeight;

        return messageId;
    }

    formatContent(content) {
        // Basic markdown support
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    async onClose() {
        // Cleanup
    }
}

class GRetrieverPlugin extends Plugin {
    async onload() {
        console.log('Loading G-Retriever Chat plugin');

        // Register view
        this.registerView(
            VIEW_TYPE_CHAT,
            (leaf) => new GRetrieverChatView(leaf, this)
        );

        // Add ribbon icon
        this.addRibbonIcon('message-square', 'Open G-Retriever Chat', () => {
            this.activateView();
        });

        // Add command
        this.addCommand({
            id: 'open-g-retriever-chat',
            name: 'Open G-Retriever Chat',
            callback: () => {
                this.activateView();
            }
        });

        // Register styles
        this.registerStyles();
    }

    async getBackendUrl() {
        // Try to read config file
        try {
            const fs = require('fs');
            const os = require('os');
            const path = require('path');

            const configPath = path.join(os.homedir(), '.g-retriever-config.json');
            const configData = fs.readFileSync(configPath, 'utf8');
            const config = JSON.parse(configData);

            return config.url || 'http://localhost:5000';
        } catch (error) {
            console.warn('Could not read config, using default port 5000');
            return 'http://localhost:5000';
        }
    }

    registerStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .g-retriever-chat-container {
                display: flex;
                flex-direction: column;
                height: 100%;
                padding: 10px;
            }

            .g-retriever-chat-box {
                flex: 1;
                overflow-y: auto;
                margin-bottom: 10px;
                padding: 10px;
                border: 1px solid var(--background-modifier-border);
                border-radius: 4px;
            }

            .g-retriever-message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
                max-width: 85%;
            }

            .g-retriever-user {
                background: var(--interactive-accent);
                color: var(--text-on-accent);
                margin-left: auto;
                text-align: right;
            }

            .g-retriever-assistant {
                background: var(--background-secondary);
                margin-right: auto;
            }

            .g-retriever-system {
                background: var(--background-modifier-border);
                color: var(--text-muted);
                font-size: 0.9em;
                text-align: center;
                max-width: 100%;
            }

            .g-retriever-debug-section {
                background: var(--background-primary-alt);
                border-left: 3px solid var(--interactive-accent);
                padding: 10px;
                margin: 10px 0;
                font-family: var(--font-monospace);
                font-size: 0.85em;
                border-radius: 4px;
            }

            .g-retriever-debug-line {
                padding: 2px 0;
                color: var(--text-muted);
            }

            .g-retriever-debug-line.query {
                font-weight: bold;
                color: var(--text-normal);
                margin-bottom: 8px;
            }

            .g-retriever-debug-line.step {
                color: var(--text-accent);
            }

            .g-retriever-debug-line.debug {
                color: var(--text-faint);
                font-size: 0.9em;
            }

            .g-retriever-debug-line.sources {
                color: var(--text-normal);
                font-weight: 500;
                margin-top: 8px;
            }

            .g-retriever-debug-line.error {
                color: var(--text-error);
                font-weight: bold;
            }

            .g-retriever-message-content {
                word-wrap: break-word;
            }

            .g-retriever-input-container {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            .g-retriever-input {
                width: 100%;
                padding: 8px;
                border: 1px solid var(--background-modifier-border);
                border-radius: 4px;
                background: var(--background-primary);
                color: var(--text-normal);
                resize: vertical;
                min-height: 60px;
                font-family: var(--font-interface);
            }

            .g-retriever-send-button {
                align-self: flex-end;
                padding: 8px 16px;
                background: var(--interactive-accent);
                color: var(--text-on-accent);
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }

            .g-retriever-send-button:hover {
                background: var(--interactive-accent-hover);
            }
        `;
        document.head.appendChild(style);
    }

    async activateView() {
        const { workspace } = this.app;

        let leaf = null;
        const leaves = workspace.getLeavesOfType(VIEW_TYPE_CHAT);

        if (leaves.length > 0) {
            leaf = leaves[0];
        } else {
            leaf = workspace.getRightLeaf(false);
            await leaf.setViewState({
                type: VIEW_TYPE_CHAT,
                active: true,
            });
        }

        workspace.revealLeaf(leaf);
    }

    onunload() {
        console.log('Unloading G-Retriever Chat plugin');
    }
}

module.exports = GRetrieverPlugin;
