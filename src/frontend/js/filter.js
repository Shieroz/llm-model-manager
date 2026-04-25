// Client-side search/filter for configs and storage
const Filter = (() => {
    function filterConfigs() {
        const term = Utils.getEl("configSearch").value.toLowerCase();
        document.querySelectorAll(".config-card").forEach(card => {
            card.style.display = card.textContent.toLowerCase().includes(term) ? "" : "none";
        });
    }

    function filterStorage() {
        const term = Utils.getEl("storageSearch").value.toLowerCase();
        document.querySelectorAll(".storage-repo-card").forEach(card => {
            card.style.display = card.textContent.toLowerCase().includes(term) ? "" : "none";
        });
    }

    function init() {
        Utils.getEl("configSearch").addEventListener("input", filterConfigs);
        Utils.getEl("storageSearch").addEventListener("input", filterStorage);
    }

    return { init, filterConfigs, filterStorage };
})();
