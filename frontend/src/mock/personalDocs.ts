import type { PersonalDocument, PersonalKnowledgeGraph } from "../types/agent";

// 03 Mock 数据（评审 S6）：后端文档/图谱接口未就绪时，由 client.ts 按 VITE_USE_MOCK 返回。
// 结构严格匹配 types/agent.ts 的 PersonalDocument / PersonalKnowledgeGraph。

export const mockDocuments: PersonalDocument[] = [
  {
    id: "doc-1",
    filename: "招商银行2023年报.pdf",
    size: 4_200_000,
    status: "done",
    uploadedAt: "2026-08-01T09:12:00.000Z",
    summary:
      "招商银行2023年报：涵盖零售金融、对公业务、财富管理三大板块，业务分布于华东、华南等地区，受净息差影响显著。",
  },
  {
    id: "doc-2",
    filename: "稳健股票型基金说明.txt",
    size: 12_400,
    status: "done",
    uploadedAt: "2026-08-02T14:30:00.000Z",
    summary:
      "某股票型基金：主要投资权益资产，风险等级 R3（中风险），目标客群为退休及稳健型投资者。",
  },
];

export const mockGraph: PersonalKnowledgeGraph = {
  nodes: [
    // doc-1
    { id: "n-cmb", label: "招商银行", type: "organization", sourceDocId: "doc-1" },
    { id: "n-retail", label: "零售金融", type: "product", sourceDocId: "doc-1" },
    { id: "n-corporate", label: "对公业务", type: "product", sourceDocId: "doc-1" },
    { id: "n-wealth", label: "财富管理", type: "product", sourceDocId: "doc-1" },
    { id: "n-east", label: "华东地区", type: "region", sourceDocId: "doc-1" },
    { id: "n-south", label: "华南地区", type: "region", sourceDocId: "doc-1" },
    { id: "n-rate", label: "净息差", type: "attribute", sourceDocId: "doc-1", properties: { 趋势: "收窄", 影响: "利润承压" } },
    // doc-2
    { id: "n-fund", label: "稳健股票基金", type: "product", sourceDocId: "doc-2" },
    { id: "n-equity", label: "权益资产", type: "attribute", sourceDocId: "doc-2" },
    { id: "n-risk", label: "R3 中风险", type: "attribute", sourceDocId: "doc-2" },
    { id: "n-elder", label: "退休客群", type: "attribute", sourceDocId: "doc-2" },
    // 跨文档
    { id: "n-cmb-fund", label: "招银理财", type: "organization", sourceDocId: "doc-1" },
  ],
  edges: [
    { id: "e1", source: "n-cmb", target: "n-retail", label: "has business", sourceDocId: "doc-1" },
    { id: "e2", source: "n-cmb", target: "n-corporate", label: "has business", sourceDocId: "doc-1" },
    { id: "e3", source: "n-cmb", target: "n-wealth", label: "has business", sourceDocId: "doc-1" },
    { id: "e4", source: "n-cmb", target: "n-cmb-fund", label: "subsidiary", sourceDocId: "doc-1" },
    { id: "e5", source: "n-retail", target: "n-east", label: "distribution area", sourceDocId: "doc-1" },
    { id: "e6", source: "n-retail", target: "n-south", label: "distribution area", sourceDocId: "doc-1" },
    { id: "e7", source: "n-corporate", target: "n-east", label: "distribution area", sourceDocId: "doc-1" },
    { id: "e8", source: "n-wealth", target: "n-rate", label: "affected by", sourceDocId: "doc-1" },
    { id: "e9", source: "n-cmb-fund", target: "n-fund", label: "issues", sourceDocId: "doc-2" },
    { id: "e10", source: "n-fund", target: "n-equity", label: "invests in", sourceDocId: "doc-2" },
    { id: "e11", source: "n-fund", target: "n-risk", label: "risk level", sourceDocId: "doc-2" },
    { id: "e12", source: "n-fund", target: "n-elder", label: "targets", sourceDocId: "doc-2" },
    { id: "e13", source: "n-wealth", target: "n-fund", label: "recommends", sourceDocId: "doc-2" },
  ],
};
