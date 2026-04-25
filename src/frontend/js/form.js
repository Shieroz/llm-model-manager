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
        return select.value || "latest";
    }

    async function handleFormSubmit(e) {
        e.preventDefault();
        const btn = Utils.getEl("submitBtn");
        btn.disabled = true;
        btn.classList.add("opacity-50");
        Utils.showStatus("Checking storage and initializing...", "warning");

        const payload = {
            hf_repo: Utils.getEl("hf_repo").value,
            quant: Utils.getEl("quant").value,
            mmproj: Utils.getEl("mmproj").value,
            symlink_name: Utils.getEl("symlink_name").value,
            original_name: Utils.getEl("original_name").value,
            parameters: Utils.getEl("parameters").value,
            revision: getSelectedRevision()
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

        Utils.getEl("setupForm").reset();
        Utils.getEl("original_name").value = "";
        Utils.getEl("quant").innerHTML = '<option value="">Paste repo above to load quants...</option>';
        Utils.getEl("quant").disabled = true;
        Utils.getEl("mmproj_container").classList.add("hidden");
        Utils.getEl("commit_select").innerHTML = '<option value="">Select a commit...</option>';
        Utils.getEl("commit_sha").value = "";
        Utils.getEl("commit_info").classList.add("hidden");

        const btn = Utils.getEl("submitBtn");
        btn.textContent = "Provision Model";
        btn.className = "flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition shadow";

        const clearBtn = Utils.getEl("clearBtn");
        clearBtn.textContent = "Clear";
        clearBtn.className = "bg-gray-600 hover:bg-gray-500 text-white py-2 px-4 rounded transition shadow";

        paramInput.value = JSON.stringify(defaultParams, null, 2);
        Utils.hideStatus();

        paramInput.classList.remove("border-red-500", "focus:ring-red-500", "border-green-500", "focus:ring-green-500");
        paramInput.classList.add("border-gray-600", "focus:ring-blue-500");
    }

    return { init, resetForm, getSelectedRevision };
})();
