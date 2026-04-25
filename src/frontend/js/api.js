// API communication functions
const Api = (() => {
    async function rpcMode(enabled) {
        const res = await fetch("/api/rpc_mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled })
        });
        return res;
    }

    async function fetchQuants(repo) {
        const res = await fetch(`/api/quants?repo=${encodeURIComponent(repo)}`);
        if (!res.ok) throw new Error("Repo not found");
        return res.json();
    }

    async function fetchCommits(repo) {
        const res = await fetch(`/api/commits?repo=${encodeURIComponent(repo)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
    }

    async function fetchLocalModels() {
        const res = await fetch("/api/models");
        return res.json();
    }

    async function setupConfig(payload) {
        const res = await fetch("/api/setup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        return res.json();
    }

    async function deleteConfig(name) {
        const res = await fetch(`/api/configs/${name}`, { method: "DELETE" });
        return res;
    }

    async function deleteRevision(repo, sha) {
        const res = await fetch("/api/revisions/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ repo, revision: sha })
        });
        return res.json();
    }

    return {
        rpcMode,
        fetchQuants,
        fetchCommits,
        fetchLocalModels,
        setupConfig,
        deleteConfig,
        deleteRevision
    };
})();
