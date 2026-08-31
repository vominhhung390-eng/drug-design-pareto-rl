import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const artifactToolPath = process.env.ARTIFACT_TOOL_MJS;
if (!artifactToolPath) {
  throw new Error("Set ARTIFACT_TOOL_MJS to the installed @oai/artifact-tool artifact_tool.mjs path.");
}
const { SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactToolPath).href);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "outputs", "019f5762-58d7-7670-9168-54fe5fbeb2b3");
const PER_SEED_CSV = path.join(OUT, "per_seed_metrics.csv");
const OUTPUT_XLSX = path.join(OUT, "主实验_Table2_Table3_10seeds.xlsx");

const methods = ["Ours (V4)", "POLYGON", "REINVENT4", "DrugEx v2", "MO-LSO", "GraphPareto-NSGA-II"];
const metrics = [
  { key: "validity", col: "G", label: "Validity↑", digits: 4 },
  { key: "uniqueness", col: "H", label: "Uniqueness↑", digits: 4 },
  { key: "novelty", col: "I", label: "Novelty↑", digits: 4 },
  { key: "diversity", col: "J", label: "Diversity↑", digits: 4 },
  { key: "hypervolume", col: "K", label: "HV↑", digits: 4 },
  { key: "igd_plus", col: "L", label: "IGD+↓", digits: 4 },
  { key: "pareto_size", col: "M", label: "Pareto size↑", digits: 1 },
  { key: "dual_at_6", col: "N", label: "Dual@6↑", digits: 4 },
  { key: "quality_pass", col: "O", label: "Quality pass↑", digits: 4 },
  { key: "alert_free", col: "P", label: "Alert-free↑", digits: 4 },
  { key: "scaffold_diversity", col: "Q", label: "Scaffold diversity↑", digits: 4 },
  { key: "qc_hypervolume", col: "R", label: "QC-HV↑", digits: 4 },
  { key: "qc_dual_at_6", col: "S", label: "QC-Dual@6↑", digits: 4 },
  { key: "qc_dual_at_7", col: "T", label: "QC-Dual@7↑", digits: 4 },
  { key: "qc_best_min", col: "U", label: "QC best-min↑", digits: 3 },
];

const generationKeys = [
  "validity",
  "uniqueness",
  "novelty",
  "diversity",
  "hypervolume",
  "igd_plus",
  "pareto_size",
  "dual_at_6",
];
const qualityKeys = [
  "quality_pass",
  "alert_free",
  "scaffold_diversity",
  "qc_hypervolume",
  "qc_dual_at_6",
  "qc_dual_at_7",
  "qc_best_min",
];

const colors = {
  navy: "#17324D",
  teal: "#167D8D",
  tealDark: "#0F6674",
  tealPale: "#E6F4F5",
  bluePale: "#EEF5FA",
  grayPale: "#F5F7F9",
  gray: "#D7DEE5",
  text: "#1F2933",
  muted: "#5C6773",
  white: "#FFFFFF",
};

function calcRow(methodIndex, metricKey) {
  const metricIndex = metrics.findIndex((metric) => metric.key === metricKey);
  return 2 + methodIndex * metrics.length + metricIndex;
}

function displayFormula(methodIndex, metricKey) {
  const metric = metrics.find((item) => item.key === metricKey);
  const row = calcRow(methodIndex, metricKey);
  const numberPattern = metric.digits === 1 ? "0.0" : metric.digits === 3 ? "0.000" : "0.0000";
  return `=TEXT(Calculations!C${row},"${numberPattern}")&" ± "&TEXT(Calculations!D${row},"${numberPattern}")`;
}

function applyBaseTypography(sheet, usedRange) {
  sheet.showGridLines = false;
  usedRange.format.font = { name: "Aptos", size: 10, color: colors.text };
  usedRange.format.verticalAlignment = "center";
}

function stylePaperTable(sheet, lastColumn, title, headers, metricKeys) {
  const lastRow = 3 + methods.length;
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: colors.navy,
    font: { name: "Aptos Display", size: 16, bold: true, color: colors.white },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    rowHeight: 31,
  };

  sheet.getRange(`A3:${lastColumn}3`).values = [headers];
  sheet.getRange(`A3:${lastColumn}3`).format = {
    fill: colors.teal,
    font: { name: "Aptos", size: 10, bold: true, color: colors.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    rowHeight: 34,
    borders: { preset: "all", style: "thin", color: colors.white },
  };

  const methodValues = methods.map((method) => [method]);
  sheet.getRange(`A4:A${lastRow}`).values = methodValues;
  for (let methodIndex = 0; methodIndex < methods.length; methodIndex += 1) {
    const formulaRow = metricKeys.map((key) => displayFormula(methodIndex, key));
    sheet.getRangeByIndexes(3 + methodIndex, 1, 1, metricKeys.length).formulas = [formulaRow];
  }

  sheet.getRange(`A4:${lastColumn}${lastRow}`).format = {
    font: { name: "Aptos", size: 10, color: colors.text },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    rowHeight: 27,
    borders: {
      insideHorizontal: { style: "thin", color: colors.gray },
      bottom: { style: "medium", color: colors.navy },
    },
  };
  sheet.getRange(`A4:A${lastRow}`).format.horizontalAlignment = "left";
  sheet.getRange(`A4:${lastColumn}4`).format = {
    fill: colors.tealPale,
    font: { name: "Aptos", size: 10, bold: true, color: colors.tealDark },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    rowHeight: 28,
    borders: {
      top: { style: "medium", color: colors.teal },
      bottom: { style: "thin", color: colors.gray },
    },
  };
  sheet.getRange("A4").format.horizontalAlignment = "left";
  for (let row = 5; row <= lastRow; row += 2) {
    sheet.getRange(`A${row}:${lastColumn}${row}`).format.fill = colors.grayPale;
  }

  const noteRow = lastRow + 2;
  sheet.getRange(`A${noteRow}:${lastColumn}${noteRow}`).merge();
  sheet.getRange(`A${noteRow}`).values = [["所有数值均为均值 ± 标准差，基于 10 个种子（seed 42–51），每个 seed 生成预算为 10,240。"]];
  sheet.getRange(`A${noteRow}:${lastColumn}${noteRow}`).format = {
    fill: colors.bluePale,
    font: { name: "Aptos", size: 10, italic: true, color: colors.muted },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
    rowHeight: 25,
  };
  sheet.getRange(`A${noteRow + 1}:${lastColumn}${noteRow + 1}`).merge();
  sheet.getRange(`A${noteRow + 1}`).values = [["↑ 表示越大越好；↓ 表示越小越好。活性为统一 EGFR/VEGFR2 预测器输出的 pActivity，不代表实验测定值。"]];
  sheet.getRange(`A${noteRow + 1}:${lastColumn}${noteRow + 1}`).format = {
    font: { name: "Aptos", size: 9, color: colors.muted },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
    rowHeight: 23,
  };

  sheet.getRange("A:A").format.columnWidth = 18;
  sheet.getRange(`B:${lastColumn}`).format.columnWidth = 18;
  sheet.freezePanes.freezeRows(3);
  applyBaseTypography(sheet, sheet.getRange(`A1:${lastColumn}${noteRow + 1}`));
  sheet.getRange(`A1:${lastColumn}1`).format.font = {
    name: "Aptos Display",
    size: 16,
    bold: true,
    color: colors.white,
  };
  sheet.getRange(`A3:${lastColumn}3`).format.font = {
    name: "Aptos",
    size: 10,
    bold: true,
    color: colors.white,
  };
  sheet.getRange(`A4:${lastColumn}4`).format.font = {
    name: "Aptos",
    size: 10,
    bold: false,
    color: colors.text,
  };
  sheet.getRange("A4").format.font = {
    name: "Aptos",
    size: 10,
    bold: true,
    color: colors.tealDark,
  };
  sheet.getRange(`A${noteRow}:${lastColumn}${noteRow}`).format.font = {
    name: "Aptos",
    size: 10,
    italic: true,
    color: colors.muted,
  };
  sheet.getRange(`A${noteRow + 1}:${lastColumn}${noteRow + 1}`).format.font = {
    name: "Aptos",
    size: 9,
    color: colors.muted,
  };
}

await fs.mkdir(OUT, { recursive: true });
const perSeedCsv = await fs.readFile(PER_SEED_CSV, "utf8");

const importedWorkbook = await Workbook.fromCSV(perSeedCsv, { sheetName: "Per-seed metrics" });
const importedPerSeed = importedWorkbook.worksheets.getItem("Per-seed metrics");
const perSeedValues = importedPerSeed.getUsedRange().values;

const workbook = Workbook.create();
const table2 = workbook.worksheets.add("Table 2 - Generation");
const table3 = workbook.worksheets.add("Table 3 - Quality");
const definitions = workbook.worksheets.add("Definitions");
const calculations = workbook.worksheets.add("Calculations");
const perSeed = workbook.worksheets.add("Per-seed metrics");
perSeed.getRange("A1:U61").values = perSeedValues;

// Formula-based aggregation sheet. Each method occupies ten consecutive rows in Per-seed metrics.
const calcHeaders = [["Method", "Metric", "Mean", "SD", "n", "Source range"]];
calculations.getRange("A1:F1").values = calcHeaders;
const calcValues = [];
const meanFormulas = [];
const sdFormulas = [];
for (let methodIndex = 0; methodIndex < methods.length; methodIndex += 1) {
  const start = 2 + methodIndex * 10;
  const end = start + 9;
  for (const metric of metrics) {
    const source = `'Per-seed metrics'!${metric.col}${start}:${metric.col}${end}`;
    calcValues.push([methods[methodIndex], metric.key, null, null, 10, source]);
    meanFormulas.push([`=AVERAGE(${source})`]);
    sdFormulas.push([`=STDEV.S(${source})`]);
  }
}
const calcLastRow = calcValues.length + 1;
calculations.getRange(`A2:F${calcLastRow}`).values = calcValues;
calculations.getRange(`C2:C${calcLastRow}`).formulas = meanFormulas;
calculations.getRange(`D2:D${calcLastRow}`).formulas = sdFormulas;
calculations.getRange(`A1:F${calcLastRow}`).format.borders = { preset: "all", style: "thin", color: colors.gray };
calculations.getRange("A1:F1").format = {
  fill: colors.navy,
  font: { name: "Aptos", size: 10, bold: true, color: colors.white },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 25,
};
calculations.getRange(`C2:D${calcLastRow}`).format.numberFormat = "0.000000";
calculations.getRange(`A2:F${calcLastRow}`).format.rowHeight = 20;
calculations.getRange("A:A").format.columnWidth = 18;
calculations.getRange("B:B").format.columnWidth = 23;
calculations.getRange("C:E").format.columnWidth = 13;
calculations.getRange("F:F").format.columnWidth = 36;
calculations.freezePanes.freezeRows(1);
applyBaseTypography(calculations, calculations.getRange(`A1:F${calcLastRow}`));
calculations.getRange("A1:F1").format.font = {
  name: "Aptos",
  size: 10,
  bold: true,
  color: colors.white,
};

stylePaperTable(
  table2,
  "I",
  "Table 2：主要生成性能",
  ["Method", ...generationKeys.map((key) => metrics.find((metric) => metric.key === key).label)],
  generationKeys,
);
stylePaperTable(
  table3,
  "H",
  "Table 3：质量约束后的结果",
  ["Method", ...qualityKeys.map((key) => metrics.find((metric) => metric.key === key).label)],
  qualityKeys,
);

// Method cell remains left aligned and highlights the proposed method.
table2.getRange("A4").format.horizontalAlignment = "left";
table3.getRange("A4").format.horizontalAlignment = "left";

const definitionRows = [
  ["实验协议", "当前方法 Ours (V4) 与五个正式基线；共同训练集；seed 42–51；每个 seed 预算 10,240。", "", ""],
  ["活性空间", "EGFR 与 VEGFR2 预测 pActivity；归一化采用 clip((pActivity − 3)/7, 0, 1)。", "", ""],
  ["Metric", "Definition", "Denominator / space", "Direction"],
  ["Validity", "生成记录中可解析、活性分数有限的有效分子比例。", "全部 10,240 个生成记录", "↑"],
  ["Uniqueness", "有效分子经去盐、RDKit 规范化后，唯一 SMILES 的比例。", "有效分子", "↑"],
  ["Novelty", "唯一有效规范 SMILES 未出现在规范化共同训练集中的比例。", "唯一有效分子", "↑"],
  ["Diversity", "1 − 平均成对 Tanimoto 相似度；ECFP4/Morgan radius=2、2048 bit；每个 run 固定抽样 min(2,000, n)。", "唯一有效分子抽样", "↑"],
  ["HV", "二维归一化活性空间中的超体积，参考点为 (0, 0)。", "唯一有效分子的 Pareto 前沿", "↑"],
  ["IGD+", "每个 run 的归一化 Pareto 前沿到统一经验参考前沿的 IGD+；参考前沿由 6 方法 × 10 seeds 合并后取非支配解（9 点）。", "统一归一化活性空间", "↓"],
  ["Pareto size", "非支配唯一有效分子数量；相同目标分数但不同分子仍分别计数。", "唯一有效分子", "↑"],
  ["Dual@6", "EGFR 与 VEGFR2 预测 pActivity 均不低于 6 的比例。", "唯一有效分子", "↑"],
  ["Quality pass", "同时满足 QED≥0.60、SA≤4.0、无 PAINS/Brenk 警报及 Lipinski 规则。", "唯一有效分子", "↑"],
  ["Alert-free", "不命中 PAINS/Brenk 结构警报的比例。", "唯一有效分子", "↑"],
  ["Scaffold diversity", "非空唯一 Bemis–Murcko scaffold 数 / 唯一有效分子数。", "唯一有效分子", "↑"],
  ["QC-HV", "仅对 Quality-pass 子集计算的归一化二维超体积，参考点为 (0, 0)。", "Quality-pass 分子", "↑"],
  ["QC-Dual@6", "Quality-pass 子集中，两个靶点预测 pActivity 均不低于 6 的比例。", "Quality-pass 分子", "↑"],
  ["QC-Dual@7", "Quality-pass 子集中，两个靶点预测 pActivity 均不低于 7 的比例。", "Quality-pass 分子", "↑"],
  ["QC best-min", "Quality-pass 子集中 max[min(EGFR pActivity, VEGFR2 pActivity)]。", "Quality-pass 分子；pActivity", "↑"],
];
definitions.getRange(`A1:D${definitionRows.length}`).values = definitionRows;
definitions.showGridLines = false;
definitions.getRange("A1:D1").merge();
definitions.getRange("A1").values = [["主实验指标定义与统计口径"]];
definitions.getRange("A1:D1").format = {
  fill: colors.navy,
  font: { name: "Aptos Display", size: 16, bold: true, color: colors.white },
  horizontalAlignment: "left",
  verticalAlignment: "center",
  rowHeight: 31,
};
definitions.getRange("A2:D2").merge();
definitions.getRange("A2").values = [[definitionRows[0][1]]];
definitions.getRange("A3:D3").merge();
definitions.getRange("A3").values = [[definitionRows[1][1]]];
definitions.getRange("A2:D3").format = {
  fill: colors.bluePale,
  font: { name: "Aptos", size: 10, color: colors.muted },
  wrapText: true,
  verticalAlignment: "center",
  rowHeight: 25,
};
// Re-write tabular definitions below protocol block.
const tableDefinitions = definitionRows.slice(2);
definitions.getRange(`A5:D${4 + tableDefinitions.length}`).values = tableDefinitions;
definitions.getRange("A5:D5").format = {
  fill: colors.teal,
  font: { name: "Aptos", size: 10, bold: true, color: colors.white },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 27,
};
definitions.getRange(`A6:D${4 + tableDefinitions.length}`).format = {
  font: { name: "Aptos", size: 10, color: colors.text },
  verticalAlignment: "center",
  wrapText: true,
  rowHeight: 35,
  borders: { preset: "all", style: "thin", color: colors.gray },
};
definitions.getRange("A:A").format.columnWidth = 22;
definitions.getRange("B:B").format.columnWidth = 72;
definitions.getRange("C:C").format.columnWidth = 32;
definitions.getRange("D:D").format.columnWidth = 12;
definitions.freezePanes.freezeRows(5);
applyBaseTypography(definitions, definitions.getRange(`A1:D${4 + tableDefinitions.length}`));
definitions.getRange("A1:D1").format.font = {
  name: "Aptos Display",
  size: 16,
  bold: true,
  color: colors.white,
};
definitions.getRange("A5:D5").format.font = {
  name: "Aptos",
  size: 10,
  bold: true,
  color: colors.white,
};

// Raw per-seed audit sheet.
const perSeedUsed = perSeed.getUsedRange();
perSeed.showGridLines = false;
perSeed.getRange("A1:U1").format = {
  fill: colors.navy,
  font: { name: "Aptos", size: 9, bold: true, color: colors.white },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  rowHeight: 34,
};
perSeed.getRange("A2:U61").format = {
  font: { name: "Aptos", size: 9, color: colors.text },
  verticalAlignment: "center",
  rowHeight: 19,
  borders: { preset: "all", style: "thin", color: colors.gray },
};
perSeed.getRange("G2:U61").format.numberFormat = "0.000000";
perSeed.getRange("A:A").format.columnWidth = 17;
perSeed.getRange("B:F").format.columnWidth = 14;
perSeed.getRange("G:U").format.columnWidth = 16;
perSeed.freezePanes.freezeRows(1);
applyBaseTypography(perSeed, perSeedUsed);
perSeed.getRange("A1:U1").format.font = {
  name: "Aptos",
  size: 9,
  bold: true,
  color: colors.white,
};

// Workbook QA: inspect formulas and rendered regions before export.
const inspection = await workbook.inspect({
  kind: "region,formula",
  sheetId: "Table 2 - Generation",
  range: "A1:I12",
  maxChars: 8000,
});
await fs.writeFile(path.join(OUT, "workbook_inspection.txt"), inspection.ndjson ?? String(inspection), "utf8");

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 5000,
});
await fs.writeFile(path.join(OUT, "formula_error_scan.txt"), errorScan.ndjson ?? String(errorScan), "utf8");

for (const [sheetName, fileName] of [
  ["Table 2 - Generation", "preview_table2.png"],
  ["Table 3 - Quality", "preview_table3.png"],
]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1.4, format: "png" });
  await fs.writeFile(path.join(OUT, fileName), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUTPUT_XLSX);
console.log(JSON.stringify({ output: OUTPUT_XLSX, sheets: 5 }, null, 2));
