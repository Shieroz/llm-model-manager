// Form handling and validation
const Form = (() => {
    const defaultParams = AppState.getDefaultParams();
    let mmprojSelect, symlinkInput, paramInput;

    function init() {
        mmprojSelect = Utils.getEl("mmproj");
        symlinkInput = Utils.getEl("symlink_name");
        paramInput = Utils.getEl("parameters");
        paramInput.value = JSON.stringify(defaultParams, null, 2);

        mmprojSelect.addEventListener("change", updateMmprojJSON);
        symlinkInput.addEventListener("input", updateMmprojJSON);

        Utils.getEl("mtp_head").addEventListener("change", onMtpHeadChange);
        Utils.getEl("mtp_n_max").addEventListener("input", onNmaxChange);
        symlinkInput.addEventListener("input", reinjectMtpPath);

        paramInput.addEventListener("input", validateJson);
        paramInput.addEventListener("blur", autoFormatJson);

        Utils.getEl("setupForm").addEventListener("submit", handleFormSubmit);
        Utils.getEl("commit_select").addEventListener("change", onCommitSelectChange);
        Utils.getEl("commit_sha").addEventListener("input", onCommitShaInput);
    }

    function validateJson() {
        if (!paramInput.value.trim()) {
            paramInput.classList.remove("border-red-500", "focus:ring-red-500", "border-green-500", "focus:ring-green-500");
            paramInput.classList.add("border-gray-600", "focus:ring-blue-500");
            return;
        }
        try {
            JSON.parse(paramInput.value);
            paramInput.classList.remove("border-red-500", "focus:ring-red-500", "border-gray-600", "focus:ring-blue-500");
            paramInput.classList.add("border-green-500", "focus:ring-green-500");
        } catch (e) {
            paramInput.classList.remove("border-green-500", "focus:ring-green-500", "border-gray-600", "focus:ring-blue-500");
            paramInput.classList.add("border-red-500", "focus:ring-red-500");
        }
    }

    function autoFormatJson() {
        if (!paramInput.value.trim()) return;
        try {
            const parsed = JSON.parse(paramInput.value);
            paramInput.value = JSON.stringify(parsed, null, 2);
        } catch (e) {}
    }

    function updateMmprojJSON() {
        const val = mmprojSelect.value;
        const sym = symlinkInput.value || "model";
        try {
            const parsed = JSON.parse(paramInput.value);
            if (val) {
                parsed["mmproj"] = `/models/served/${sym}-mmproj-${val.toUpperCase()}.gguf`;
            } else {
                delete parsed["mmproj"];
            }
            paramInput.value = JSON.stringify(parsed, null, 2);
            paramInput.dispatchEvent(new Event("input"));
        } catch (e) {}
    }

    // Heuristic: a grafted-MTP repo carries "mtp" as a token in its name (e.g.
    // unsloth/Qwen3.6-27B-MTP-GGUF) and bakes the MTP head into the quant file, so we
    // enable speculative decoding automatically for those.
    function repoLooksGrafted() {
        const repo = (Utils.getEl("hf_repo").value || "");
        return /(^|[\/\-_.])mtp([\/\-_.]|$)/i.test(repo);
    }

    function nMax() {
        return parseInt(Utils.getEl("mtp_n_max").value, 10) || 4;
    }

    function servedDraftPath() {
        const sym = symlinkInput.value || "model";
        return `/models/served/${sym}-mtp-head.gguf`;
    }

    // Rewrite the spec-* flags (and model-draft path) in the params JSON.
    function applySpec(active, withHead) {
        try {
            const parsed = JSON.parse(paramInput.value);
            if (active) {
                parsed["spec-type"] = "draft-mtp";
                parsed["spec-draft-n-max"] = nMax();
                if (withHead) parsed["model-draft"] = servedDraftPath();
                else delete parsed["model-draft"];
            } else {
                delete parsed["spec-type"];
                delete parsed["spec-draft-n-max"];
                delete parsed["model-draft"];
            }
            paramInput.value = JSON.stringify(parsed, null, 2);
            paramInput.dispatchEvent(new Event("input"));
        } catch (e) {}
    }

    // Configure the MTP box from a fetched /api/quants payload. Grafted builds inject
    // the spec flags immediately; separate-head repos expose a draft-head dropdown.
    function applyMtpUI(data) {
        const headSel = Utils.getEl("mtp_head");
        const heads = (data && data.heads) || [];
        headSel.innerHTML = '<option value="">None</option>';
        heads.forEach(h => {
            const opt = document.createElement("option");
            opt.value = h.name;
            opt.textContent = `${h.name} (${h.size_str})`;
            headSel.appendChild(opt);
        });
        const grafted = repoLooksGrafted();
        const hasHeads = heads.length > 0;
        Utils.getEl("mtp_box").classList.toggle("hidden", !(grafted || hasHeads));
        Utils.getEl("mtp_grafted_note").classList.toggle("hidden", !grafted);
        Utils.getEl("mtp_head_row").classList.toggle("hidden", !hasHeads);
        Utils.getEl("mtp_nmax_row").classList.toggle("hidden", !grafted);
        if (grafted) applySpec(true, false);
    }

    function hideMtp() {
        Utils.getEl("mtp_box").classList.add("hidden");
    }

    function onMtpHeadChange() {
        const head = Utils.getEl("mtp_head").value;
        const grafted = repoLooksGrafted();
        Utils.getEl("mtp_nmax_row").classList.toggle("hidden", !(grafted || !!head));
        applySpec(grafted || !!head, !!head);
    }

    function onNmaxChange() {
        try {
            const parsed = JSON.parse(paramInput.value);
            if (parsed["spec-type"] === "draft-mtp") {
                parsed["spec-draft-n-max"] = nMax();
                paramInput.value = JSON.stringify(parsed, null, 2);
                paramInput.dispatchEvent(new Event("input"));
            }
        } catch (e) {}
    }

    // Keep the model-draft symlink path aligned with the symlink name.
    function reinjectMtpPath() {
        if (Utils.getEl("mtp_head").value) applySpec(true, true);
    }

    // Reflect MTP state from the current params JSON (used on edit / load). The head
    // dropdown value is restored by the caller before this runs.
    function syncMtpFromParams() {
        try {
            const parsed = JSON.parse(paramInput.value);
            const on = parsed["spec-type"] === "draft-mtp";
            if (on && parsed["spec-draft-n-max"] != null) {
                Utils.getEl("mtp_n_max").value = parsed["spec-draft-n-max"];
            }
            Utils.getEl("mtp_nmax_row").classList.toggle("hidden", !on);
        } catch (e) {}
    }

    function onCommitSelectChange() {
        const shaInput = Utils.getEl("commit_sha");
        if (this.value === "latest") {
            shaInput.value = "";
        }
    }

    function onCommitShaInput() {
        const select = Utils.getEl("commit_select");
        if (this.value.trim()) {
            select.value = "custom";
        }
    }

    function getSelectedRevision() {
        const select = Utils.getEl("commit_select");
        const shaInput = Utils.getEl("commit_sha");
        if (shaInput.value.trim()) {
            return shaInput.value.trim();
        }
        if (select.value === "latest") {
            return AppState.selectedBranch || "main";
        }
        return select.value || "latest";
    }

    async function handleFormSubmit(e) {
        e.preventDefault();
        const btn = Utils.getEl("submitBtn");
        btn.disabled = true;
        btn.classList.add("opacity-50");
        Utils.showStatus("Checking storage and initializing...", "warning");

        const headRow = Utils.getEl("mtp_head_row");
        const payload = {
            hf_repo: Utils.getEl("hf_repo").value,
            quant: Utils.getEl("quant").value,
            mmproj: Utils.getEl("mmproj").value,
            symlink_name: Utils.getEl("symlink_name").value,
            original_name: Utils.getEl("original_name").value,
            parameters: Utils.getEl("parameters").value,
            revision: getSelectedRevision(),
            mtp_head: headRow.classList.contains("hidden") ? "" : Utils.getEl("mtp_head").value
        };

        try {
            const data = await Api.setupConfig(payload);
            if (data.status && data.status.includes("Warning")) {
                Utils.showStatus(data.status, "warning");
            } else if (data.status) {
                Utils.showStatus(data.status, "success");
            } else {
                Utils.showStatus("Error: " + (data.detail || "Invalid parameters"), "error");
                btn.disabled = false;
                btn.classList.remove("opacity-50");
                return;
            }
            setTimeout(() => {
                resetForm();
                btn.disabled = false;
                btn.classList.remove("opacity-50");
                LocalModels.fetch();
            }, 3500);
        } catch (e) {
            Utils.showStatus("Network error occurred.", "error");
            btn.disabled = false;
            btn.classList.remove("opacity-50");
        }
    }

    function resetForm() {
        const title = Utils.getEl("formTitle");
        title.textContent = "Deploy New Config";
        title.className = "text-2xl font-bold mb-4 text-blue-400";

        Utils.getEl("hf_repo").value = "";
        Utils.getEl("setupForm").reset();
        Utils.getEl("original_name").value = "";
        Utils.getEl("quant").innerHTML = '<option value="">Paste repo above to load quants...</option>';
        Utils.getEl("quant").disabled = true;
        Utils.getEl("mmproj_container").classList.add("hidden");
        Utils.getEl("commit_select").innerHTML = '<option value="">Enter a repo first</option>';
        Utils.getEl("commit_select").disabled = true;
        Utils.getEl("commit_sha").value = "";
        Utils.getEl("commit_sha").disabled = true;
        Utils.getEl("commit_sha").placeholder = "Enter a repo first";
        Utils.getEl("commit_info").classList.add("hidden");
        Utils.getEl("branch_select").innerHTML = '<option value="">Enter a repo first</option>';
        Utils.getEl("branch_select").disabled = true;
        AppState.selectedBranch = "main";
        AppState.lastCommitsStr = "";
        Utils.getEl("quant").innerHTML = '<option value="">Enter a repo first</option>';
        Utils.getEl("quant").disabled = true;
        Utils.getEl("symlink_name").disabled = true;
        Utils.getEl("mmproj").disabled = true;
        AppState.selectedBranch = "main";

        const btn = Utils.getEl("submitBtn");
        btn.textContent = "Provision Model";
        btn.className = "flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition shadow";

        const clearBtn = Utils.getEl("clearBtn");
        clearBtn.textContent = "Clear";
        clearBtn.className = "bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded transition shadow";

        paramInput.value = JSON.stringify(defaultParams, null, 2);
        Utils.getEl("mtp_box").classList.add("hidden");
        Utils.getEl("mtp_grafted_note").classList.add("hidden");
        Utils.getEl("mtp_head_row").classList.add("hidden");
        Utils.getEl("mtp_nmax_row").classList.add("hidden");
        Utils.getEl("mtp_head").innerHTML = '<option value="">None</option>';
        Utils.getEl("mtp_n_max").value = 4;
        Utils.hideStatus();

        paramInput.classList.remove("border-red-500", "focus:ring-red-500", "border-green-500", "focus:ring-green-500");
        paramInput.classList.add("border-gray-600", "focus:ring-blue-500");
    }

    return { init, resetForm, getSelectedRevision, syncMtpFromParams, applyMtpUI, hideMtp };
})();
