"""
QA Engine nâng cao cho Phase 1.
Trả lời câu hỏi của tài xế về luật giao thông Việt Nam,
biển báo, cấm đỗ, tốc độ, camera phạt nguội...

Architecture:
  1. Rule-based lookup (nhanh, chính xác cho câu hỏi chuẩn)
  2. Keyword matching (fuzzy)  
  3. Semantic search với FAISS + sentence-transformers (optional)
  4. LLM fallback (optional - Gemini API / local model)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class QAResult:
    """Kết quả trả lời câu hỏi."""
    answer: str
    confidence: float       # 0–1
    source: str             # "rule", "keyword", "semantic", "llm"
    matched_key: Optional[str] = None


# =====================================================================
# Knowledge Base — Luật giao thông Việt Nam
# =====================================================================

TRAFFIC_KNOWLEDGE: dict = {
    # Biển cấm
    "bien_cam": {
        "cấm đỗ xe": "Biển P.130 – Cấm đỗ xe. Xe không được đỗ tại khu vực này.",
        "cấm dừng xe": "Biển P.131a/b – Cấm dừng xe và đỗ xe. Xe không được dừng hay đỗ.",
        "cấm rẽ trái": "Biển P.102a – Cấm rẽ trái. Xe phải đi thẳng hoặc rẽ phải.",
        "cấm rẽ phải": "Biển P.102b – Cấm rẽ phải. Xe phải đi thẳng hoặc rẽ trái.",
        "cấm vào": "Biển P.101 – Cấm đi vào. Xe không được vào đường này.",
        "cấm quay đầu": "Biển P.124a – Cấm quay đầu xe. Không được quay đầu.",
        "cấm vượt": "Biển P.115 – Cấm vượt. Không được vượt xe khác (ngoại trừ xe 2 bánh).",
        "cấm xe tải": "Biển P.105 – Cấm xe tải. Xe tải không được vào đường này.",
        "cấm xe máy": "Biển P.106c – Cấm mô tô và xe máy đi vào.",
    },
    # Tốc độ
    "toc_do": {
        "tốc độ trong khu dân cư": "Tối đa 40 km/h trong khu vực đông dân cư (Nghị định 46/2016).",
        "tốc độ nội thành": "Tối đa 50–60 km/h tùy tuyến đường nội thành.",
        "tốc độ quốc lộ": "Xe con: tối đa 90 km/h. Xe tải: 70–80 km/h trên quốc lộ.",
        "tốc độ cao tốc": "Xe con: 120 km/h tối đa. Tối thiểu 60 km/h trên đường cao tốc.",
        "tốc độ học sinh": "30 km/h gần trường học vào giờ vào/tan học.",
    },
    # Camera phạt nguội
    "camera": {
        "camera phạt nguội": (
            "Camera phạt nguội (camera giao thông) ghi lại vi phạm tốc độ, "
            "vượt đèn đỏ, đi sai làn. Phạt qua bưu điện hoặc tại CSGT."
        ),
        "vượt đèn đỏ phạt bao nhiêu": (
            "Vượt đèn đỏ: xe máy 800k–1 triệu, xe ô tô 3–5 triệu đồng, "
            "tước bằng 1–3 tháng (Nghị định 123/2021)."
        ),
    },
    # Ưu tiên đường
    "uu_tien": {
        "xe ưu tiên": (
            "Xe ưu tiên gồm: xe cứu thương, xe cứu hỏa, xe cảnh sát, "
            "xe hộ tống. Các phương tiện khác phải nhường đường."
        ),
        "vạch kẻ đường": (
            "Vạch liền: không được vượt. Vạch đứt: có thể vượt khi an toàn. "
            "Vạch đôi liền: tuyệt đối không vượt."
        ),
    },
    # Phạm vi đường
    "duong": {
        "đường một chiều": "Đường một chiều: chỉ được đi theo chiều mũi tên. Không đi ngược chiều.",
        "đường ưu tiên": "Biển I.407 – Đường ưu tiên. Các đường phụ nhường đường cho bạn.",
        "đường cấm": "Đường chỉ cho phép một số loại xe nhất định. Kiểm tra biển báo cụ thể.",
    },
    # Hỏi thăm đường
    "qa_chung": {
        "có camera không": "Dựa trên dữ liệu bản đồ, tuyến đường này có/không có camera phạt nguội. Nên tuân thủ tốc độ quy định.",
        "có cấm gì không": "Đang phân tích biển báo trên đường... Vui lòng kiểm tra màn hình hiển thị biển báo.",
        "đường này đỗ xe được không": "Quan sát biển báo cấm đỗ trên đường. Không có biển P.130 = được đỗ (nhưng cần tắt đèn cảnh báo).",
    },
}

# Flatten knowledge base thành danh sách (question, answer) pairs
_FLAT_KB: List[Tuple[str, str]] = []
for _cat in TRAFFIC_KNOWLEDGE.values():
    for _q, _a in _cat.items():
        _FLAT_KB.append((_q, _a))


class TrafficQAEngine:
    """
    Công cụ Q&A về giao thông Việt Nam.

    Level 1: Exact/keyword match (không cần GPU)
    Level 2: Semantic similarity (cần sentence-transformers)
    Level 3: LLM fallback (cần API key)
    """

    def __init__(self, use_semantic: bool = False, use_llm: bool = False) -> None:
        self._use_semantic = use_semantic
        self._use_llm = use_llm
        self._embedder = None
        self._index = None
        self._index_answers: List[str] = []

        if use_semantic:
            self._init_semantic()

    def _init_semantic(self) -> None:
        """Khởi tạo FAISS index từ knowledge base."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            self._embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            questions = [q for q, _ in _FLAT_KB]
            self._index_answers = [a for _, a in _FLAT_KB]

            embeddings = self._embedder.encode(questions)
            d = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(d)
            self._index.add(embeddings.astype("float32"))
        except ImportError:
            self._use_semantic = False

    def answer(self, question: str) -> QAResult:
        """Trả lời câu hỏi về giao thông."""
        q = question.lower().strip()

        # Level 1: Exact match
        result = self._rule_lookup(q)
        if result:
            return result

        # Level 2: Keyword match  
        result = self._keyword_match(q)
        if result:
            return result

        # Level 3: Semantic search
        if self._use_semantic and self._index is not None:
            result = self._semantic_search(question)
            if result and result.confidence > 0.5:
                return result

        # Level 4: LLM fallback
        if self._use_llm:
            return self._llm_answer(question)

        return QAResult(
            answer=(
                "Xin lỗi, tôi chưa có thông tin về câu hỏi này. "
                "Vui lòng tham khảo Luật Giao thông đường bộ hoặc hỏi cảnh sát giao thông."
            ),
            confidence=0.0,
            source="fallback",
        )

    def _rule_lookup(self, q: str) -> Optional[QAResult]:
        """Tìm exact match trong knowledge base."""
        for question_key, answer in _FLAT_KB:
            if question_key in q or q in question_key:
                return QAResult(
                    answer=answer,
                    confidence=1.0,
                    source="rule",
                    matched_key=question_key,
                )
        return None

    def _keyword_match(self, q: str) -> Optional[QAResult]:
        """Tìm theo keyword."""
        keyword_map = {
            ("cấm đỗ", "đỗ xe", "dừng đỗ"): "cấm đỗ xe",
            ("cấm rẽ trái", "không rẽ trái"): "cấm rẽ trái",
            ("cấm rẽ phải", "không rẽ phải"): "cấm rẽ phải",
            ("cấm vào", "không vào", "cấm đi vào"): "cấm vào",
            ("tốc độ", "chạy bao nhiêu", "km/h"): "tốc độ nội thành",
            ("camera", "phạt nguội", "bắn tốc độ"): "camera phạt nguội",
            ("vượt đèn đỏ", "đèn đỏ", "phạt đèn"): "vượt đèn đỏ phạt bao nhiêu",
            ("cấm quay đầu", "quay đầu"): "cấm quay đầu",
            ("vạch kẻ", "vạch đường", "vượt vạch"): "vạch kẻ đường",
            ("xe ưu tiên", "cứu thương", "cứu hỏa"): "xe ưu tiên",
            ("cấm vượt", "không vượt"): "cấm vượt",
            ("cao tốc",): "tốc độ cao tốc",
            ("quốc lộ",): "tốc độ quốc lộ",
        }

        for keywords, lookup_key in keyword_map.items():
            if any(kw in q for kw in keywords):
                for question_key, answer in _FLAT_KB:
                    if question_key == lookup_key:
                        return QAResult(
                            answer=answer,
                            confidence=0.85,
                            source="keyword",
                            matched_key=lookup_key,
                        )
        return None

    def _semantic_search(self, question: str) -> Optional[QAResult]:
        """Tìm kiếm semantic với FAISS."""
        try:
            import numpy as np
            emb = self._embedder.encode([question]).astype("float32")
            distances, indices = self._index.search(emb, k=1)
            dist = float(distances[0][0])
            idx = int(indices[0][0])
            # Chuyển L2 distance thành confidence (0–1)
            confidence = max(0.0, 1.0 - dist / 10.0)
            return QAResult(
                answer=self._index_answers[idx],
                confidence=confidence,
                source="semantic",
            )
        except Exception:
            return None

    def _llm_answer(self, question: str) -> QAResult:
        """LLM fallback - sử dụng Gemini hoặc local LLM."""
        try:
            import google.generativeai as genai
            api_key = __import__("os").environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("No GEMINI_API_KEY")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Bạn là chuyên gia luật giao thông Việt Nam. "
                "Trả lời ngắn gọn, chính xác (1-3 câu) câu hỏi sau:\n"
                f"Câu hỏi: {question}"
            )
            response = model.generate_content(prompt)
            return QAResult(
                answer=response.text.strip(),
                confidence=0.7,
                source="llm",
            )
        except Exception as e:
            return QAResult(
                answer=f"Không thể trả lời qua LLM: {e}",
                confidence=0.0,
                source="llm_error",
            )

    def query_from_sign(self, sign_type: str) -> Optional[str]:
        """Tra cứu nghĩa của biển báo từ tên/mã biển."""
        sign_map = {
            "P.130": "Cấm đỗ xe",
            "P.131": "Cấm dừng và đỗ xe",
            "P.102a": "Cấm rẽ trái",
            "P.102b": "Cấm rẽ phải",
            "P.101": "Cấm đi vào",
            "P.124a": "Cấm quay đầu xe",
            "P.115": "Cấm vượt xe",
            "no_parking": "Cấm đỗ xe",
            "no_entry": "Cấm vào",
            "no_left_turn": "Cấm rẽ trái",
            "no_right_turn": "Cấm rẽ phải",
            "no_u_turn": "Cấm quay đầu",
            "speed_limit": "Giới hạn tốc độ",
            "stop": "Dừng lại",
        }
        key = sign_type.lower().replace(" ", "_")
        for k, v in sign_map.items():
            if k.lower() in key or key in k.lower():
                result = self.answer(v)
                return result.answer
        return None
