// Utility functions
const Utils = (() => {
    function debounce(fn, delay) {
        return function (...args) {
            clearTimeout(AppState.repoDebounceTimer);
            AppState.repoDebounceTimer = setTimeout(() => fn.apply(this, args), delay);
        };
    }

    function formatBytes(bytes) {
        if (bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB", "TB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }

    function shortSha(sha, len) {
        return sha ? sha.substring(0, len || 12) : "";
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function getEl(id) {
        return document.getElementById(id);
    }

    function showStatus(msg, type) {
        const el = getEl("statusMsg");
        const cls = type === "error" ? "text-red-400" : type === "warning" ? "text-yellow-400 font-bold" : "text-green-400";
        el.className = `text-sm mt-2 ${cls} block`;
        el.textContent = msg;
    }

    function hideStatus() {
        const el = getEl("statusMsg");
        el.className = "hidden";
    }

    function formatFileSize(sizeStr) {
        return sizeStr;
    }

    function hasRpc(conf) {
        return !!(conf.params && conf.params.rpc);
    }

    return {
        debounce,
        formatBytes,
        shortSha,
        escapeHtml,
        getEl,
        showStatus,
        hideStatus,
        formatFileSize,
        hasRpc
    };
})();
