const moduleInfo = {
  density: {
    title: "Density profile",
    subtitle: "Quantify accumulation, depletion, and layering along one coordinate.",
    fields: [
      ["coordRange", "Coordinate range", "[-1.0, 30.0]"],
      ["coordBins", "Coordinate bins", "310"],
      ["species", "Oxygen species", "OH-, H2O"],
      ["normalization", "Normalization", "number_density"],
      ["prefix", "Output prefix", "waterint_density"],
    ],
  },
  "oh-orientation": {
    title: "OH orientation",
    subtitle: "Resolve O-H orientation as a function of interfacial position.",
    fields: [
      ["coordRange", "Coordinate range", "[-1.0, 8.0]"],
      ["coordBins", "Coordinate bins", "180"],
      ["species", "Oxygen species", "OH-, H2O, H3O+"],
      ["angleRange", "Angle range", "[0.0, 180.0]"],
      ["angleBins", "Angle bins", "180"],
      ["prefix", "Output prefix", "waterint_oh_orientation"],
    ],
  },
  hbond: {
    title: "H-bond topology",
    subtitle: "Compare donor and acceptor coordination environments.",
    fields: [
      ["species", "Oxygen species", "OH-, H2O, H3O+"],
      ["ooCutoff", "O-O cutoff", "3.5"],
      ["angleMin", "D-H-A angle min", "150.0"],
      ["pbc", "PBC flags", "[true, true, false]"],
      ["prefix", "Output prefix", "waterint_hbond"],
    ],
  },
  sfg: {
    title: "SFG spectrum",
    subtitle: "Estimate an interfacial O-H vibrational response.",
    fields: [
      ["dtPs", "dt_ps", "0.005"],
      ["lagPs", "lag_ps", "0.995"],
      ["pbc", "PBC flags", "[true, true, false]"],
      ["window", "Window z1,z2", "[0.0, 8.0]"],
      ["nzeros", "FT zero padding", "2000"],
      ["prefix", "Output prefix", "waterint_sfg"],
    ],
  },
};

const examplePresets = {
  density: {
    baseDir: "v0/example/mgo_density",
    trajectory: "../shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz",
    format: "xyz",
    systemCell: "[10.0, 10.0, 10.0]",
    typeMap: "",
    fields: {
      coordRange: "[0.0, 10.0]",
      coordBins: "20",
      species: "O2-, OH-, H2O, H3O+",
      normalization: "number_density",
      prefix: "ui_density_oxygen_species",
    },
  },
  "oh-orientation": {
    baseDir: "v0/example/mgo_oh_orientation",
    trajectory: "../shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz",
    format: "lammpstrj",
    systemCell: "auto",
    typeMap: "1: H\n2: O\n3: Mg",
    fields: {
      coordRange: "[0.0, 5.0]",
      coordBins: "10",
      species: "OH-, H2O, H3O+",
      angleRange: "[0.0, 180.0]",
      angleBins: "18",
      prefix: "ui_oh_orientation",
    },
  },
  hbond: {
    baseDir: "v0/example/mgo_hbond",
    trajectory: "../shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz",
    format: "lammpstrj",
    systemCell: "auto",
    typeMap: "1: H\n2: O",
    fields: {
      species: "H2O",
      ooCutoff: "3.5",
      angleMin: "150.0",
      pbc: "[false, false, false]",
      prefix: "ui_hbond_topology",
    },
  },
  sfg: {
    baseDir: "v0/example/mgo_sfg",
    trajectory: "../shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz",
    format: "lammpstrj",
    systemCell: "auto",
    typeMap: "1: H\n2: O",
    fields: {
      dtPs: "0.001",
      lagPs: "0.003",
      pbc: "[true, true, false]",
      window: "",
      nzeros: "32",
      prefix: "ui_sfg_small_oh",
    },
  },
};

const state = { jobId: null, pollTimer: null };

const $ = (id) => document.getElementById(id);

function init() {
  $("module").addEventListener("change", renderModule);
  $("exampleButton").addEventListener("click", loadExamplePreset);
  $("refreshYaml").addEventListener("click", () => ($("yamlPreview").value = buildYaml()));
  $("resetButton").addEventListener("click", renderModule);
  $("runButton").addEventListener("click", runAnalysis);
  ["trajectory", "format", "maxFrames", "systemCell", "typeMap", "unitStyle", "outLength", "outMassDensity", "outFrequency"].forEach((id) => {
    $(id).addEventListener("input", () => ($("yamlPreview").value = buildYaml()));
  });
  fetch("/api/status")
    .then((r) => r.json())
    .then((data) => {
      $("baseDir").placeholder = data.cwd || "/path/to/project";
    })
    .catch(() => {});
  renderModule();
}

function renderModule() {
  const info = moduleInfo[$("module").value];
  $("moduleTitle").textContent = info.title;
  $("moduleSubtitle").textContent = info.subtitle;
  $("moduleFields").innerHTML = "";
  info.fields.forEach(([id, label, value]) => {
    const wrapper = document.createElement("label");
    wrapper.textContent = label;
    const input = document.createElement("input");
    input.id = `field-${id}`;
    input.value = value;
    input.addEventListener("input", () => ($("yamlPreview").value = buildYaml()));
    wrapper.appendChild(input);
    $("moduleFields").appendChild(wrapper);
  });
  $("yamlPreview").value = buildYaml();
}

function loadExamplePreset() {
  const preset = examplePresets[$("module").value];
  if (!preset) return;
  $("baseDir").value = preset.baseDir;
  $("trajectory").value = preset.trajectory;
  $("format").value = preset.format;
  $("systemCell").value = preset.systemCell;
  $("typeMap").value = preset.typeMap;
  $("maxFrames").value = "all";
  Object.entries(preset.fields).forEach(([id, value]) => {
    const node = $(`field-${id}`);
    if (node) node.value = value;
  });
  $("yamlPreview").value = buildYaml();
}

function field(id) {
  const node = $(`field-${id}`);
  return node ? node.value.trim() : "";
}

function listValue(text) {
  return text
    .replace("[", "")
    .replace("]", "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

function typeMapYaml() {
  return $("typeMap")
    .value.split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => `    ${line}`)
    .join("\n");
}

function yamlString(text) {
  return JSON.stringify(String(text));
}

function buildYaml() {
  const module = $("module").value;
  const outputDir = `waterint_ui_output/${module}`;
  const lines = [];
  lines.push("input:");
  lines.push(`  trajectory: ${yamlString($("trajectory").value || "path/to/trajectory.npz")}`);
  lines.push(`  format: ${$("format").value}`);
  const typeMap = typeMapYaml();
  if (typeMap) {
    lines.push("  type_map:");
    lines.push(typeMap);
  }
  lines.push(`  max_frames: ${$("maxFrames").value || "all"}`);
  lines.push("  stride: 1");
  lines.push("");
  lines.push("system:");
  lines.push(`  cell: ${$("systemCell").value || "auto"}`);
  lines.push("");
  lines.push("units:");
  lines.push(`  style: ${$("unitStyle").value}`);
  lines.push("  output:");
  lines.push(`    length: ${$("outLength").value}`);
  lines.push(`    mass_density: ${$("outMassDensity").value}`);
  lines.push("    number_density: 1/A^3");
  lines.push("    time: ps");
  lines.push(`    frequency: ${$("outFrequency").value}`);
  lines.push("");

  if (module !== "sfg") {
    lines.push("selection:");
    if (module === "density") lines.push("  mode: oxygen_species");
    lines.push(`  oxygen_species: [${listValue(field("species")).join(", ")}]`);
    lines.push("  oxygen_symbol: O");
    lines.push("  hydrogen_symbol: H");
    lines.push("  oh_cutoff: 1.25");
    lines.push("");
  }

  if (module === "density" || module === "oh-orientation") {
    lines.push("coordinate:");
    lines.push("  mode: relative_to_slab");
    lines.push("  axis: z");
    lines.push("  reference:");
    lines.push("    type: slab_surface");
    lines.push("    species: [Mg]");
    lines.push("    surface: max");
    lines.push(`  range: ${field("coordRange")}`);
    lines.push(`  bins: ${field("coordBins")}`);
    lines.push("");
  }

  if (module === "density") {
    lines.push("normalization:");
    lines.push(`  type: ${field("normalization")}`);
    if (field("normalization") === "mass_density") {
      lines.push("  masses_amu:");
      lines.push("    O2-: 15.999");
      lines.push("    OH-: 17.007");
      lines.push("    H2O: 18.015");
      lines.push("    H3O+: 19.023");
    }
  } else if (module === "oh-orientation") {
    lines.push("angle:");
    lines.push("  vector_mode: oh_bond");
    lines.push(`  range: ${field("angleRange")}`);
    lines.push(`  bins: ${field("angleBins")}`);
    lines.push("  axis_sign: 1");
    lines.push("");
    lines.push("normalization:");
    lines.push("  type: counts_per_frame");
  } else if (module === "hbond") {
    lines.push("hbond:");
    lines.push(`  oo_cutoff: ${field("ooCutoff")}`);
    lines.push(`  dha_angle_min: ${field("angleMin")}`);
    lines.push("  h_acceptor_cutoff: null");
    lines.push("  max_acceptors_per_hydrogen: true");
    lines.push(`  pbc: ${field("pbc") || "[true, true, false]"}`);
    lines.push("  classes_by_species:");
    lines.push("    OH-: [DAAA, DAA, DA, AAA, AA, A, other]");
    lines.push("    H2O: [DDAA, DDA, DAA, DA, AA, A, other]");
    lines.push("    H3O+: [DDDA, DDD, DDA, DD, DA, D, other]");
  } else if (module === "sfg") {
    lines.push("sfg:");
    lines.push("  mode: trajectory");
    lines.push("  hydrogen_symbol: H");
    lines.push("  oxygen_symbol: O");
    lines.push(`  dt_ps: ${field("dtPs")}`);
    lines.push(`  lag_ps: ${field("lagPs")}`);
    lines.push(`  pbc: ${field("pbc") || "[true, true, false]"}`);
    lines.push("  z_ref0: 0.0");
    const windowVals = listValue(field("window"));
    if (windowVals.length >= 2) {
      lines.push("  window:");
      lines.push("    mode: 1");
      lines.push(`    z1: ${windowVals[0]}`);
      lines.push(`    z2: ${windowVals[1]}`);
      lines.push("    ramp: 0.5");
      lines.push("    flip: true");
    }
    lines.push("  mu_mode: full");
    lines.push("  symmetrize: true");
    lines.push(`  nzeros: ${field("nzeros") || "2000"}`);
  }

  lines.push("");
  lines.push("output:");
  lines.push(`  directory: ${outputDir}`);
  lines.push(`  prefix: ${field("prefix") || `waterint_${module}`}`);
  lines.push("  plot: true");
  return lines.join("\n");
}

async function runAnalysis() {
  setStatus("running", "Submitting job...");
  $("artifactGrid").innerHTML = "";
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        module: $("module").value,
        base_dir: $("baseDir").value,
        config_yaml: $("yamlPreview").value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to submit job");
    state.jobId = data.job_id;
    pollJob();
  } catch (error) {
    setStatus("failed", error.message);
  }
}

async function pollJob() {
  if (!state.jobId) return;
  const response = await fetch(`/api/jobs/${state.jobId}`);
  const job = await response.json();
  if (!response.ok) {
    setStatus("failed", job.error || "Could not read job");
    return;
  }
  setStatus(job.status, job.error || job.message);
  renderArtifacts(job);
  if (job.status === "queued" || job.status === "running") {
    state.pollTimer = setTimeout(pollJob, 1000);
  }
}

function setStatus(status, text) {
  const badge = $("statusBadge");
  badge.className = `badge ${status}`;
  badge.textContent = status;
  $("statusText").textContent = text;
}

function renderArtifacts(job) {
  const grid = $("artifactGrid");
  grid.innerHTML = "";
  (job.artifacts || []).forEach((artifact) => {
    const item = document.createElement("div");
    item.className = "artifact";
    const url = `/api/jobs/${job.id}/artifacts/${artifact.id}`;
    if (artifact.kind === "image") {
      const img = document.createElement("img");
      img.src = url;
      img.alt = artifact.name;
      item.appendChild(img);
    }
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.textContent = artifact.name;
    item.appendChild(link);
    const meta = document.createElement("p");
    meta.textContent = `${artifact.label} - ${artifact.path}`;
    item.appendChild(meta);
    grid.appendChild(item);
  });
}

init();
