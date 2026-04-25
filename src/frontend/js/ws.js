// WebSocket connection management
const Ws = (() => {
    let ws = null;
    let reconnectTimer = null;
    const RECONNECT_DELAY = 2000;

    function connect() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.host;
        ws = new WebSocket(`${protocol}//${host}/ws`);

        ws.onopen = () => {
            const el = Utils.getEl("wsStatus");
            el.innerHTML = '<span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> Live Sync Active';
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "update") {
                    handleUpdate(msg);
                }
            } catch (e) {
                console.error("WebSocket message parse error:", e);
            }
        };

        ws.onclose = () => {
            const el = Utils.getEl("wsStatus");
            el.innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span> Disconnected. Retrying...';
            reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
    }

    function handleUpdate(msg) {
        const newConfigStr = JSON.stringify(msg.data);
        const newRpcMode = msg.rpc_mode || false;

        // Skip render if both configs and RPC toggle state are identical
        if (newConfigStr === AppState.lastConfigStr && newRpcMode === AppState.isRpcMode) return;

        // Prevent WebSocket from rubber-banding the RPC toggle during an API call
        const rpcToggle = Utils.getEl("rpcToggle");
        if (!rpcToggle.disabled) {
            AppState.isRpcMode = newRpcMode;
            rpcToggle.checked = AppState.isRpcMode;
        }

        const prevDownloading = AppState.currentConfigs
            .filter(c => c.status === "downloading")
            .map(c => c.name);

        AppState.currentConfigs = msg.data;
        AppState.lastConfigStr = newConfigStr;
        Render.renderConfigs();
        Filter.filterConfigs();

        const nowDownloading = AppState.currentConfigs
            .filter(c => c.status === "downloading")
            .map(c => c.name);
        const downloadFinished = prevDownloading.some(name => !nowDownloading.includes(name));

        if (downloadFinished) {
            LocalModels.fetch();
        }
    }

    function disconnect() {
        if (ws) {
            ws.close();
            ws = null;
        }
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    }

    return { connect, disconnect };
})();
