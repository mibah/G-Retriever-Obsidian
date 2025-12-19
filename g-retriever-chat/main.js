const { Plugin, ItemView, WorkspaceLeaf, Notice } = require('obsidian');

const VIEW_TYPE_CHAT = "g-retriever-chat-view";

class GRetrieverChatView extends ItemView {
    constructor(leaf) {
        super(leaf);
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

        // Add initial message
        this.addMessage('system', 'G-Retriever ready! Ask me anything about your notes.');
    }

    async sendMessage() {
        const question = this.input.value.trim();
        if (!question) return;

        this.addMessage('user', question);
        this.input.value = '';

        // Show loading
        const loadingId = this.addMessage('assistant', 'Thinking...');

        try {
            // Call Python backend
            const response = await this.queryBackend(question);

            // Update with real response
            this.updateMessage(loadingId, response.answer);

            // Show sources
            if (response.subgraph_nodes && response.subgraph_nodes.length > 0) {
                const sources = response.subgraph_nodes.slice(0, 5).join(', ');
                this.addMessage('system', `Sources: ${sources}`, 'sources');
            }
        } catch (error) {
            this.updateMessage(loadingId, `Error: ${error.message}`);
            new Notice('Error communicating with G-Retriever backend');
        }
    }

    async queryBackend(question) {
        // Call local Python API
        const response = await fetch('http://localhost:5001/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) {
            throw new Error('Backend not responding. Is the server running?');
        }

        return await response.json();
    }

    addMessage(role, content, className = '') {
        const messageDiv = this.chatBox.createDiv({
            cls: `g-retriever-message g-retriever-${role} ${className}`
        });

        const contentDiv = messageDiv.createDiv({ cls: 'g-retriever-message-content' });
        contentDiv.textContent = content;

        this.chatBox.scrollTop = this.chatBox.scrollHeight;

        return messageDiv.id = `msg-${Date.now()}`;
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

        // Register view
        this.registerView(
            VIEW_TYPE_CHAT,
            (leaf) => new GRetrieverChatView(leaf)
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
    }

    async activateView() {
        const { workspace } = this.app;

        let leaf = null;
        const leaves = workspace.getLeavesOfType(VIEW_TYPE_CHAT);

        if (leaves.length > 0) {
            // View already exists, reveal it
            leaf = leaves[0];
        } else {
            // Create new view in right sidebar
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
