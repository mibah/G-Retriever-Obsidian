const { Plugin, ItemView, Notice, requestUrl } = require('obsidian');
const path = require('path');
const fs = require('fs');
const os = require('os');

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

        // Status indicator
        const statusBar = container.createDiv({ cls: 'g-retriever-status-bar' });
        this.statusBar = statusBar;
        this.updateStatus('checking');

        // Chat History
        const chatBox = container.createDiv({ cls: 'g-retriever-chat-box' });
        this.chatBox = chatBox;

        // Input Area
        const inputContainer = container.createDiv({ cls: 'g-retriever-input-container' });

        // Wrapper für das Textarea
        const textareaWrapper = inputContainer.createDiv({ cls: 'g-retriever-textarea-wrapper' });
        const input = textareaWrapper.createEl('textarea', {
            cls: 'g-retriever-input',
            attr: {
                placeholder: 'Ask a question about your notes...',
                rows: '3'
            }
        });
        this.input = input;

        // Wrapper für den Button
        const buttonWrapper = inputContainer.createDiv({ cls: 'g-retriever-button-wrapper' });
        const sendButton = buttonWrapper.createEl('button', {
            text: 'Send',
            cls: 'g-retriever-send-button'
        });

//                const inputContainer = container.createDiv({ cls: 'g-retriever-input-container' });
//
//        const input = inputContainer.createEl('textarea', {
//            cls: 'g-retriever-input',
//            attr: {
//                placeholder: 'Ask a question about your notes...',
//                rows: '3'
//            }
//        });
//        this.input = input;
//
//        const sendButton = inputContainer.createEl('button', {
//            text: 'Send',
//            cls: 'g-retriever-send-button'
//        });

        sendButton.addEventListener('click', () => this.sendMessage());
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Check backend status
        await this.checkBackendStatus();
    }

    async checkBackendStatus() {
        const apiUrl = await this.plugin.getApiUrl();
        
        if (!apiUrl) {
            this.updateStatus('error');
            this.addMessage('system', 
                'Backend not found. Please start the G-Retriever server first.\n\n' +
                'Run: python api_server.py\n\n' +
                'See the plugin README for setup instructions.');
            return false;
        }

        try {
            const response = await requestUrl({
                url: `${apiUrl}/health`,
                method: 'GET'
            });

            if (response.status === 200) {
                this.updateStatus('connected');
                this.addMessage('system', 'Connected to G-Retriever! Ask me anything about your notes.');
                return true;
            }
        } catch (error) {
            this.updateStatus('error');
            this.addMessage('system', 
                `Cannot connect to backend at ${apiUrl}\n\n` +
                'Please make sure the server is running:\n' +
                'python api_server.py');
            return false;
        }

        return false;
    }

    updateStatus(status) {
        this.statusBar.empty();
        
        const indicator = this.statusBar.createSpan({ cls: 'status-indicator' });
        const text = this.statusBar.createSpan({ cls: 'status-text' });

        switch(status) {
            case 'connected':
                indicator.addClass('status-connected');
                text.textContent = 'Connected';
                break;
            case 'error':
                indicator.addClass('status-error');
                text.textContent = 'Not Connected';
                break;
            case 'checking':
                indicator.addClass('status-checking');
                text.textContent = 'Checking...';
                break;
        }
    }

    async sendMessage() {
        const question = this.input.value.trim();
        if (!question) return;

        this.addMessage('user', question);
        this.input.value = '';

        const loadingId = this.addMessage('assistant', '🤔 Thinking...');

        try {
            const apiUrl = await this.plugin.getApiUrl();
            
            const response = await requestUrl({
                url: `${apiUrl}/query`,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });

            const result = response.json;
            this.updateMessage(loadingId, result.answer);
            
            if (result.subgraph_nodes && result.subgraph_nodes.length > 0) {
                const sources = result.subgraph_nodes.slice(0, 5).join(', ');
                this.addMessage('system', `📚 Sources: ${sources}`, 'sources');
            }
        } catch (error) {
            this.updateMessage(loadingId, `❌ Error: ${error.message}`);
            new Notice('Error communicating with G-Retriever. Is the server running?');
        }
    }

    addMessage(role, content, className = '') {
        const messageDiv = this.chatBox.createDiv({
            cls: `g-retriever-message g-retriever-${role} ${className}`
        });
        
        const contentDiv = messageDiv.createDiv({ cls: 'g-retriever-message-content' });
        contentDiv.textContent = content;
        
        this.chatBox.scrollTop = this.chatBox.scrollHeight;
        
        messageDiv.id = `msg-${Date.now()}-${Math.random()}`;
        return messageDiv.id;
    }

    updateMessage(id, content) {
        const messageDiv = this.chatBox.querySelector(`#${id}`);
        if (messageDiv) {
            const contentDiv = messageDiv.querySelector('.g-retriever-message-content');
            contentDiv.textContent = content;
        }
    }

    async onClose() {
        // Cleanup
    }
}

class GRetrieverPlugin extends Plugin {
    async onload() {
        console.log('Loading G-Retriever Chat plugin');

        this.registerView(
            VIEW_TYPE_CHAT,
            (leaf) => new GRetrieverChatView(leaf, this)
        );

        this.addRibbonIcon('message-square', 'Open G-Retriever Chat', () => {
            this.activateView();
        });

        this.addCommand({
            id: 'open-g-retriever-chat',
            name: 'Open G-Retriever Chat',
            callback: () => {
                this.activateView();
            }
        });
    }

    async getApiUrl() {
        // Try to read config file
        const configPath = path.join(os.homedir(), '.g-retriever-config.json');
        
        try {
            if (fs.existsSync(configPath)) {
                const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
                return config.url;
            }
        } catch (error) {
            console.error('Error reading config:', error);
        }

        // Fallback to default
        return 'http://localhost:5000';
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
