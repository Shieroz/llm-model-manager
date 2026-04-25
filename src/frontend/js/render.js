// UI rendering functions
const Render = (() => {
    function renderConfigs() {
        const list = Utils.getEl("configList");
        const filteredConfigs = AppState.currentConfigs.filter(conf => {
            return AppState.isRpcMode ? Utils.hasRpc(conf) : !Utils.hasRpc(conf);
        });

        if (filteredConfigs.length === 0) {
            list.innerHTML = `<p class="text-gray-500 italic">No matching configs found for ${AppState.isRpcMode ? "RPC" : "Local"} mode.</p>`;
            return;
        }

        // Nuke the container if the empty state text is present
        if (list.querySelector("p.italic")) { list.innerHTML = ""; }

        const currentNames = filteredConfigs.map(c => c.name);
        Array.from(list.children).forEach(child => {
            if (child.dataset && child.dataset.name && !currentNames.includes(child.dataset.name)) {
                child.remove();
            }
        });

        AppState.currentConfigs.forEach(conf => {
            const cardHTML = buildConfigCard(conf);
            const borderColor = getBorderColor(conf);

            let card = list.querySelector(`.config-card[data-name="${Utils.escapeHtml(conf.name)}"]`);
            if (!card) {
                card = document.createElement("div");
                card.dataset.name = conf.name;
                list.appendChild(card);
            }

            const newClassName = `config-card bg-gray-800 p-4 rounded-lg shadow border ${borderColor} flex flex-col`;
            if (card.className !== newClassName) card.className = newClassName;
            if (card.innerHTML !== cardHTML) card.innerHTML = cardHTML;
        });
    }

    function getBorderColor(conf) {
        const { status } = conf;
        if (status === "downloading") return "border-yellow-600";
        if (status === "error") return "border-red-600";
        if (status === "missing") return "border-orange-600";
        return "border-gray-700 hover:border-blue-500";
    }

    function buildConfigCard(conf) {
        const { status, name, quant, revision, repo, params, progress_str, error_msg } = conf;
        const isDownloading = status === "downloading";
        const isError = status === "error";
        const isMissing = status === "missing";

        const statusBadge = buildStatusBadge(conf);
        const progressUI = isDownloading ? buildProgressUI(conf) : "";
        const paramsHtml = buildParamsHtml(params);
        const buttons = buildButtons(conf);

        const revisionBadge = revision
            ? `<span class="text-xs bg-gray-700 text-gray-400 px-2 py-0.5 rounded border border-gray-600 font-mono tracking-wider shadow-sm" title="Commit: ${Utils.escapeHtml(revision)}">${Utils.shortSha(revision)}...</span>`
            : "";

        return `
            <div class="flex justify-between items-start">
                <div class="flex-1 min-w-0">
                    <h3 class="text-lg font-bold text-white flex flex-wrap items-center gap-2">
                        <span class="truncate">${Utils.escapeHtml(name)}</span>
                        <span class="text-xs bg-blue-900 text-blue-300 px-2 py-0.5 rounded border border-blue-700 font-mono tracking-wider shadow-sm">${Utils.escapeHtml(quant)}</span>
                        ${revisionBadge}
                        ${statusBadge}
                    </h3>
                    <p class="text-xs text-gray-500 mb-3 font-mono mt-1 truncate">Repo: ${Utils.escapeHtml(repo)}</p>
                    <div class="flex flex-wrap mt-2">${paramsHtml}</div>
                </div>
                ${buttons}
            </div>
            ${progressUI}
        `;
    }

    function buildStatusBadge(conf) {
        const { status } = conf;
        if (status === "downloading") {
            return '<span class="text-xs bg-yellow-900 text-yellow-300 px-2 py-0.5 rounded border border-yellow-700 animate-pulse">Downloading...</span>';
        }
        if (status === "error") {
            return '<span class="text-xs bg-red-900 text-red-300 px-2 py-0.5 rounded border border-red-700">Error</span>';
        }
        if (status === "missing") {
            return '<span class="text-xs bg-orange-900 text-orange-300 px-2 py-0.5 rounded border border-orange-700 animate-pulse">Missing Files</span>';
        }
        return "";
    }

    function buildProgressUI(conf) {
        const { name, progress_str: p } = conf;
        if (!p) return "";
        const isExpanded = AppState.expandedConfigs.has(name);

        return `
            <div class="mt-3 border-t border-gray-700 pt-3">
                <div class="flex justify-between items-center cursor-pointer hover:text-white text-gray-400 transition" onclick="toggleDetails('${name}')">
                    <div class="flex-1 mr-4">
                        <div class="h-2 bg-gray-700 rounded-full overflow-hidden w-full">
                            <div class="h-full bg-yellow-500 transition-all duration-300 ease-linear" style="width: ${p.percent}%"></div>
                        </div>
                    </div>
                    <span class="text-xs font-mono text-yellow-500 whitespace-nowrap">${p.percent}% ${isExpanded ? "\u25B2" : "\u25BC"}</span>
                </div>
                ${isExpanded ? `
                <div class="mt-3 flex justify-between text-xs font-mono text-gray-400 bg-gray-900 p-3 rounded border border-gray-700">
                    <div>Prog: <span class="text-gray-200">${p.downloaded} / ${p.total}</span></div>
                    <div>Spd: <span class="text-blue-400">${Utils.escapeHtml(p.speed)}</span></div>
                    <div>ETA: <span class="text-yellow-400">${Utils.escapeHtml(p.eta)}</span></div>
                </div>
                ` : ""}
            </div>
        `;
    }

    function buildParamsHtml(params) {
        return Object.entries(params).map(([k, v]) =>
            `<span class="bg-gray-900 border border-gray-600 text-xs px-2 py-1 rounded mr-1 mb-1 inline-block"><span class="text-blue-300">${Utils.escapeHtml(k)}:</span> <span class="font-mono text-gray-300">${typeof v === "object" ? JSON.stringify(v) : v}</span></span>`
        ).join("");
    }

    function buildButtons(conf) {
        const { name, status } = conf;
        const isDownloading = status === "downloading";
        const isError = status === "error";

        if (!isDownloading && isError) {
            const errorMsg = Utils.escapeHtml(conf.error_msg || "Download failed");
            return `<p class="text-xs text-red-400 mb-2 truncate max-w-[150px] cursor-help" title="${errorMsg}">${errorMsg}</p>
                <div class="flex gap-2">
                    <button onclick="editConfig('${name}')" class="bg-yellow-600 hover:bg-yellow-500 text-white text-xs py-1 px-3 rounded w-full">Edit</button>
                    <button onclick="deleteConfig('${name}')" class="bg-red-600 hover:bg-red-500 text-white text-xs py-1 px-3 rounded w-full">Delete</button>
                </div>`;
        }
        if (!isDownloading) {
            const editLabel = conf.status === "missing" ? "Re-Download" : "Edit";
            return `
                <div class="flex flex-col gap-2 ml-4 min-w-[80px]">
                    <button onclick="duplicateConfig('${name}')" class="bg-green-600 hover:bg-green-500 text-white text-xs py-1 px-3 rounded shadow transition">Duplicate</button>
                    <button onclick="editConfig('${name}')" class="bg-blue-600 hover:bg-blue-500 text-white text-xs py-1 px-3 rounded shadow transition">${editLabel}</button>
                    <button onclick="deleteConfig('${name}')" class="bg-red-600 hover:bg-red-500 text-white text-xs py-1 px-3 rounded shadow transition">Delete</button>
                </div>`;
        }
        return "";
    }

    function renderStorage() {
        const list = Utils.getEl("storageList");
        if (AppState.localModels.length === 0) {
            list.innerHTML = '<p class="text-gray-500 italic">No downloaded models found on disk.</p>';
            return;
        }
        if (list.querySelector("p.italic")) { list.innerHTML = ""; }

        const currentRepos = AppState.localModels.map(m => m.repo);
        Array.from(list.children).forEach(child => {
            if (child.dataset && child.dataset.repo && !currentRepos.includes(child.dataset.repo)) {
                child.remove();
            }
        });

        AppState.localModels.forEach(repoData => {
            const innerHTML = buildStorageRepoCard(repoData);
            let repoGroup = list.querySelector(`.storage-repo-card[data-repo="${Utils.escapeHtml(repoData.repo)}"]`);
            if (!repoGroup) {
                repoGroup = document.createElement("div");
                repoGroup.className = "storage-repo-card bg-gray-900 p-4 rounded-lg shadow border border-gray-700 mb-4";
                repoGroup.dataset.repo = repoData.repo;
                list.appendChild(repoGroup);
            }
            if (repoGroup.innerHTML !== innerHTML) {
                repoGroup.innerHTML = innerHTML;
            }
        });
    }

    function buildStorageRepoCard(repoData) {
        let innerHTML = `
            <div class="flex justify-between items-center mb-3 border-b border-gray-700 pb-2">
                <h3 class="text-md font-bold text-gray-300 truncate">${Utils.escapeHtml(repoData.repo)}</h3>
                <span class="text-xs font-mono text-gray-500 shrink-0 ml-2">${Utils.escapeHtml(repoData.size_str)}</span>
            </div>
            <div class="space-y-3">`;

        (repoData.revisions || []).forEach(rev => {
            innerHTML += buildStorageRevisionCard(repoData, rev);
        });

        innerHTML += "</div>";
        return innerHTML;
    }

    function buildStorageRevisionCard(repoData, rev) {
        const blockId = `${repoData.repo}-${rev.sha}`;
        const isExpanded = AppState.expandedStorage.has(blockId);
        const refs = (rev.refs || []).map(r =>
            `<span class="text-xs bg-blue-900 text-blue-300 px-1.5 py-0.5 rounded border border-blue-700 font-mono">${Utils.escapeHtml(r)}</span>`
        ).join(" ");

        const usedBy = (rev.used_by || []).map(name =>
            `<span class="text-xs bg-green-900 text-green-300 px-1.5 py-0.5 rounded border border-green-700 font-mono">${Utils.escapeHtml(name)}</span>`
        ).join(" ");
        const pinned = (rev.used_by || []).length > 0;

        const { quants, mmprojQuants } = extractQuants(rev.files);
        const quantBadges = quants.map(q =>
            `<span class="text-xs bg-green-900 text-green-300 px-2 py-0.5 rounded border border-green-700 font-mono">${Utils.escapeHtml(q)}</span>`
        ).join(" ");
        const mmprojBadges = mmprojQuants.map(q =>
            `<span class="text-xs bg-purple-900 text-purple-300 px-2 py-0.5 rounded border border-purple-700 font-mono">${Utils.escapeHtml(q)} vision</span>`
        ).join(" ");

        const filesHtml = isExpanded ? buildFilesList(rev.files) : "";

        const deleteBtn = pinned
            ? `<button disabled title="Pinned by: ${(rev.used_by || []).join(", ")}" class="text-xs bg-gray-700 text-gray-500 py-1 px-2 rounded shrink-0 cursor-not-allowed">Pinned</button>`
            : `<button onclick="deleteRevision('${Utils.escapeHtml(repoData.repo)}', '${Utils.escapeHtml(rev.sha)}')" class="text-xs bg-red-600 hover:bg-red-500 text-white py-1 px-2 rounded shadow transition shrink-0">Delete</button>`;

        return `
            <div class="storage-file-row bg-gray-800 p-3 rounded border border-gray-700">
                <div class="flex justify-between items-start gap-2">
                    <div class="flex flex-col gap-1.5 cursor-pointer file-expand-icon text-gray-400 flex-1 min-w-0" onclick="toggleStorageDetails('${blockId}')">
                        <div class="flex items-center gap-2 flex-wrap">
                            <span class="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded border border-gray-600 font-mono" title="${Utils.escapeHtml(rev.sha)}">${Utils.escapeHtml(rev.short_sha)}</span>
                            ${refs}
                            <span class="text-sm font-mono text-gray-300">${Utils.escapeHtml(rev.size_str)}</span>
                            <span class="text-xs">${isExpanded ? "\u25B2" : "\u25BC"}</span>
                        </div>
                        <div class="flex items-center gap-1 flex-wrap">${quantBadges}${mmprojBadges}</div>
                        ${usedBy ? `<div class="flex items-center gap-1 flex-wrap"><span class="text-[10px] uppercase tracking-wider text-gray-500">used by:</span> ${usedBy}</div>` : ""}
                    </div>
                    ${deleteBtn}
                </div>
                ${filesHtml}
            </div>`;
    }

    function extractQuants(files) {
        const quants = new Set();
        const mmprojQuants = new Set();
        files.forEach(f => {
            if (f.quant) {
                if (f.is_mmproj) {
                    mmprojQuants.add(f.quant);
                } else {
                    quants.add(f.quant);
                }
            }
        });
        return { quants: Array.from(quants), mmprojQuants: Array.from(mmprojQuants) };
    }

    function buildFilesList(files) {
        let html = `<div class="mt-3 pl-4 space-y-1 border-l-2 border-gray-700">`;
        files.forEach(f => {
            const tag = f.is_mmproj
                ? "bg-purple-900 text-purple-300 border-purple-700"
                : "bg-gray-700 text-gray-300 border-gray-600";
            html += `
                <div class="flex justify-between items-center text-xs font-mono gap-2">
                    <span class="flex items-center gap-2 min-w-0">
                        <span class="text-[10px] ${tag} px-1.5 py-0.5 rounded border shrink-0">${Utils.escapeHtml(f.quant || "?")}${f.is_mmproj ? " vision" : ""}</span>
                        <span class="text-gray-400 truncate">${Utils.escapeHtml(f.name)}</span>
                    </span>
                    <span class="text-gray-500 shrink-0">${Utils.escapeHtml(f.size_str)}</span>
                </div>`;
        });
        html += "</div>";
        return html;
    }

    return { renderConfigs, renderStorage };
})();
