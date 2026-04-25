// Main application entry point - wires all modules together
const App = (() => {
    // Expose functions that need to be called from inline HTML event handlers
    function exposeGlobals() {
        window.toggleRPCMode = toggleRPCMode;
        window.toggleDetails = toggleDetails;
        window.toggleStorageDetails = toggleStorageDetails;
        window.editConfig = editConfig;
        window.duplicateConfig = duplicateConfig;
        window.deleteConfig = deleteConfig;
        window.deleteRevision = deleteRevision;
        window.resetForm = Form.resetForm;
        window.filterConfigs = Filter.filterConfigs;
        window.filterStorage = Filter.filterStorage;
    }

    async function toggleRPCMode() {
        const toggle = Utils.getEl("rpcToggle");
        toggle.disabled = true;

        AppState.isRpcMode = toggle.checked;
        Render.renderConfigs();
        Filter.filterConfigs();

        try {
            await Api.rpcMode(toggle.checked);
        } catch (e) {
            alert("Failed to toggle RPC mode");
            toggle.checked = !toggle.checked;
            AppState.isRpcMode = toggle.checked;
            Render.renderConfigs();
            Filter.filterConfigs();
        }
        toggle.disabled = false;
    }

    function toggleDetails(name) {
        if (AppState.expandedConfigs.has(name)) {
            AppState.expandedConfigs.delete(name);
        } else {
            AppState.expandedConfigs.add(name);
        }
        Render.renderConfigs();
        Filter.filterConfigs();
    }

    function toggleStorageDetails(id) {
        if (AppState.expandedStorage.has(id)) {
            AppState.expandedStorage.delete(id);
        } else {
            AppState.expandedStorage.add(id);
        }
        Render.renderStorage();
        Filter.filterStorage();
    }

    async function editConfig(name) {
        const conf = AppState.currentConfigs.find(c => c.name === name);
        if (!conf) return;

        const title = Utils.getEl("formTitle");
        title.textContent = `Edit Config: ${name}`;
        title.className = "text-2xl font-bold mb-4 text-yellow-400";

        Utils.getEl("hf_repo").value = conf.repo;
        await Api.fetchQuants(conf.repo).then(data => populateQuants(data, false));
        await fetchCommits(conf.repo, conf.revision);
        populateRevision(conf.revision);

        Utils.getEl("quant").value = conf.quant;
        Utils.getEl("mmproj").value = conf.mmproj || "";
        Utils.getEl("symlink_name").value = conf.name;
        Utils.getEl("original_name").value = conf.name;

        const btn = Utils.getEl("submitBtn");
        btn.textContent = "Update / Re-Download";
        btn.className = "flex-1 bg-yellow-600 hover:bg-yellow-500 text-white font-bold py-2 px-4 rounded transition shadow";

        const clearBtn = Utils.getEl("clearBtn");
        clearBtn.textContent = "Cancel Edit";
        clearBtn.className = "bg-red-900 hover:bg-red-800 text-white py-2 px-4 rounded transition shadow";

        let params = JSON.parse(JSON.stringify(conf.params));
        if (conf.mmproj && params.mmproj) {
            params.mmproj = `/models/served/${conf.name}-mmproj-${conf.mmproj.toUpperCase()}.gguf`;
        }
        Utils.getEl("parameters").value = JSON.stringify(params, null, 2);
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    async function duplicateConfig(name) {
        const conf = AppState.currentConfigs.find(c => c.name === name);
        if (!conf) return;

        Form.resetForm();
        const title = Utils.getEl("formTitle");
        title.textContent = `Duplicate Config: ${name}`;
        title.className = "text-2xl font-bold mb-4 text-green-400";

        Utils.getEl("hf_repo").value = conf.repo;
        await Api.fetchQuants(conf.repo).then(data => populateQuants(data, false));
        await fetchCommits(conf.repo, conf.revision);
        populateRevision(conf.revision);

        Utils.getEl("quant").value = conf.quant;
        Utils.getEl("mmproj").value = conf.mmproj || "";
        Utils.getEl("symlink_name").value = conf.name + "-COPY";

        const btn = Utils.getEl("submitBtn");
        btn.textContent = "Provision Duplicate";
        btn.className = "flex-1 bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded transition shadow";

        const clearBtn = Utils.getEl("clearBtn");
        clearBtn.textContent = "Cancel";
        clearBtn.className = "bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded transition shadow";

        let params = JSON.parse(JSON.stringify(conf.params));
        if (conf.mmproj && params.mmproj) {
            params.mmproj = `/models/served/${conf.name}-mmproj-${conf.mmproj.toUpperCase()}.gguf`;
        }
        Utils.getEl("parameters").value = JSON.stringify(params, null, 2);
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    async function deleteConfig(name) {
        if (confirm(`Delete configuration ${name}? (Physical model files will remain on disk)`)) {
            await Api.deleteConfig(name);
            if (Utils.getEl("original_name").value === name) {
                Form.resetForm();
            }
        }
    }

    async function deleteRevision(repo, sha) {
        if (!confirm(`Delete cached revision ${Utils.shortSha(sha)} for ${repo}?\n\nBlobs unique to this revision will be removed. Shared blobs remain.`)) return;
        try {
            const data = await Api.deleteRevision(repo, sha);
            if (data && data.status === 0 || data.ok) {
                LocalModels.fetch();
                return;
            }
            if (data.status === 409 && data.detail && data.detail.used_by) {
                const names = data.detail.used_by.join(", ");
                alert(`Cannot delete revision: it is pinned by the following configs.\n\n  ${names}\n\nDelete those configs first, then retry.`);
            } else {
                const detail = data.detail || data;
                alert("Failed to delete revision: " + (typeof detail === "string" ? detail : JSON.stringify(detail)));
            }
        } catch (e) {
            console.error(e);
            alert("Network error while deleting revision.");
        }
    }

    async function fetchCommits(repo, currentSha = null) {
        const select = Utils.getEl("commit_select");
        const info = Utils.getEl("commit_info");
        try {
            const data = await Api.fetchCommits(repo);
            const newCommitsStr = JSON.stringify(data.commits || []);
            if (newCommitsStr === AppState.lastCommitsStr) return;
            AppState.lastCommitsStr = newCommitsStr;

            select.innerHTML = "";
            if (!data.commits || data.commits.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "No commits found";
                select.appendChild(opt);
                info.classList.remove("hidden");
                info.textContent = "No commits found";
                info.className = "text-xs text-yellow-500 mt-1";
                return;
            }
            data.commits.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c.sha;
                const label = Utils.shortSha(c.sha);
                const isCurrent = currentSha && c.sha === currentSha;
                const isPinned = c.pinned && !isCurrent;
                if (isCurrent) {
                    opt.textContent = `${label} (current)`;
                    opt.style.fontWeight = "bold";
                    opt.style.color = "#22c55e";
                } else if (isPinned) {
                    opt.textContent = `${label} (pinned)`;
                    opt.style.color = "#f59e0b";
                } else {
                    opt.textContent = label;
                }
                opt.title = c.message + "\n" + new Date(c.date * 1000).toLocaleString();
                select.appendChild(opt);
            });
            // Select current commit if editing, otherwise latest (first in list)
            if (currentSha) {
                const match = Array.from(select.options).find(o => o.value === currentSha);
                select.value = match ? currentSha : (select.options[0]?.value || "");
            } else {
                select.value = select.options[0]?.value || "";
            }
            info.classList.remove("hidden");
            info.textContent = data.commits.length + " commits available";
            info.className = "text-xs text-green-500 mt-1";
        } catch (e) {
            select.innerHTML = '<option value="">Error loading commits</option>';
            select.value = "";
            info.classList.remove("hidden");
            info.textContent = "Could not load commits: " + e.message;
            info.className = "text-xs text-red-500 mt-1";
            console.error("fetchCommits failed:", e);
        }
    }

    function populateRevision(rev) {
        const commitSelect = Utils.getEl("commit_select");
        const commitSha = Utils.getEl("commit_sha");
        if (!rev) {
            commitSelect.value = "latest";
            commitSha.value = "";
            return;
        }
        const match = Array.from(commitSelect.options).find(o => o.value === rev);
        if (match) {
            commitSelect.value = rev;
            commitSha.value = "";
        } else {
            commitSelect.value = "latest";
            commitSha.value = rev;
        }
    }

    async function populateQuants(data, autoFillSymlink) {
        const quantSelect = Utils.getEl("quant");
        const symInput = Utils.getEl("symlink_name");
        if (autoFillSymlink && !symInput.value) {
            symInput.value = symInput.value || data.repoName || "";
        }
        quantSelect.innerHTML = '<option value="">Scanning Hugging Face...</option>';
        quantSelect.disabled = true;

        if (data.quants.length === 0) {
            quantSelect.innerHTML = '<option value="">No GGUF quants found</option>';
            return;
        }

        quantSelect.innerHTML = "";
        data.quants.forEach(q => {
            const opt = document.createElement("option");
            opt.value = q.name;
            opt.textContent = `${q.name} (${q.size_str})`;
            quantSelect.appendChild(opt);
        });

        const largestQ4 = data.quants.find(q => q.name.includes("Q4"));
        quantSelect.value = largestQ4 ? largestQ4.name : data.quants[0].name;
        quantSelect.disabled = false;

        const mmprojContainer = Utils.getEl("mmproj_container");
        const mmprojSelect = Utils.getEl("mmproj");
        mmprojSelect.innerHTML = '<option value="">None</option>';
        if (data.mmprojs && data.mmprojs.length > 0) {
            mmprojContainer.classList.remove("hidden");
            data.mmprojs.forEach(q => {
                const opt = document.createElement("option");
                opt.value = q.name;
                opt.textContent = `${q.name} (${q.size_str})`;
                mmprojSelect.appendChild(opt);
            });
        } else {
            mmprojContainer.classList.add("hidden");
        }
    }

    function initRepoDebounce() {
        const repoInput = Utils.getEl("hf_repo");
        repoInput.addEventListener("input", Utils.debounce(async (e) => {
            const repo = e.target.value.trim();
            if (!repo.includes("/")) return;
            await Api.fetchQuants(repo).then(data => populateQuants(data, true));
            await fetchCommits(repo);
        }, 800));
    }

    return { init: () => { exposeGlobals(); Form.init(); initRepoDebounce(); } };
})();
