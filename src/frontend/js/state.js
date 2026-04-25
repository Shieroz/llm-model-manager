// Shared application state
const AppState = (() => {
    let currentConfigs = [];
    let localModels = [];
    let expandedConfigs = new Set();
    let expandedStorage = new Set();
    let repoDebounceTimer = null;
    let lastConfigStr = "";
    let lastStorageStr = "";
    let lastCommitsStr = "";
    let isRpcMode = false;

    return {
        get currentConfigs() { return currentConfigs; },
        set currentConfigs(val) { currentConfigs = val; },
        get localModels() { return localModels; },
        set localModels(val) { localModels = val; },
        get expandedConfigs() { return expandedConfigs; },
        get expandedStorage() { return expandedStorage; },
        get repoDebounceTimer() { return repoDebounceTimer; },
        set repoDebounceTimer(val) { repoDebounceTimer = val; },
        get lastConfigStr() { return lastConfigStr; },
        set lastConfigStr(val) { lastConfigStr = val; },
        get lastStorageStr() { return lastStorageStr; },
        set lastStorageStr(val) { lastStorageStr = val; },
        get lastCommitsStr() { return lastCommitsStr; },
        set lastCommitsStr(val) { lastCommitsStr = val; },
        get isRpcMode() { return isRpcMode; },
        set isRpcMode(val) { isRpcMode = val; },
        getDefaultParams() {
            return {
                "no-mmap": true,
                "fit": "on",
                "fitc": 16384,
                "fa": "on",
                "cache-type-k": "q8_0",
                "cache-type-v": "q8_0",
                "t": 8,
                "n": -1,
                "temp": 0.6,
                "top-p": 0.95,
                "top-k": 40,
                "min-p": 0
            };
        }
    };
})();
