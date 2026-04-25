// Local models / disk storage data fetching
const LocalModels = (() => {
    async function fetch() {
        try {
            const data = await Api.fetchLocalModels();
            const newStorageStr = JSON.stringify(data.models || []);
            if (newStorageStr !== AppState.lastStorageStr) {
                AppState.localModels = data.models || [];
                AppState.lastStorageStr = newStorageStr;
                Render.renderStorage();
                Filter.filterStorage();
            }
        } catch (e) {
            console.error("Failed to fetch local models", e);
        }
    }

    return { fetch };
})();
