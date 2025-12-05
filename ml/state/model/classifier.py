import numpy as np
from loguru import logger

from ml.state.constants import EPSILON, WEIGHT_COSINE, WEIGHT_EUCLIDEAN


class AdvancedPresetAnalyzer:
    """
    Acts as a container for prototypes and similarity logic.
    (Kept class name for compatibility with existing loading code,
     but striped of heavy dependencies).
    """

    def __init__(self) -> None:
        self.preset_prototypes: dict[str, np.ndarray] = {}

    def find_most_similar_preset(self, feature_vector: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """
        Compares input vector against all loaded prototypes.
        Returns: (best_preset_name, confidence_score, debug_details)
        """
        best_preset = 'FLOW'
        best_score = -1.0
        scores: dict[str, float] = {}

        if not self.preset_prototypes:
            return best_preset, 0.0, {}

        # 0. Extract Confidence
        current_confidence = feature_vector[0]

        # 0. Extract slice with Basic Emotions only
        vec_basic = feature_vector[1:6]

        for preset_name, prototype in self.preset_prototypes.items():
            if feature_vector.shape != prototype.shape:
                continue

            # 1. Cosine Similarity (BASIC EMOTIONS ONLY)
            # Original Research code used only raw emotions to estimate vector direction
            proto_basic = prototype[1:6]

            norm_input = vec_basic / (np.linalg.norm(vec_basic) + EPSILON)
            norm_proto = proto_basic / (np.linalg.norm(proto_basic) + EPSILON)
            cosine_sim = np.dot(norm_input, norm_proto)

            # 2. Euclidean Similarity (FULL VECTOR)
            # Distance metric accounts for entropy and intensity
            dist = np.linalg.norm(feature_vector - prototype)
            euclidean_sim = 1.0 / (1.0 + dist)

            # 3. Weighted Combination + Confidence Boost
            combined = (WEIGHT_COSINE * cosine_sim) + (WEIGHT_EUCLIDEAN * euclidean_sim)

            # If the camera clearly sees the face, we trust this match more.
            boosted_score = combined * (1.0 + 0.2 * current_confidence)

            final_score = float(np.clip(boosted_score, 0.0, 1.0))
            scores[preset_name] = final_score

            if final_score > best_score:
                best_score = final_score
                best_preset = preset_name

        return best_preset, best_score, scores


class AdvancedEmotionClassifier:
    """
    Main entry point for State Logic.
    Takes basic probabilities -> Generates Features -> Predicts State.
    """

    def __init__(self, analyzer: AdvancedPresetAnalyzer) -> None:
        self.analyzer = analyzer

        # Expected input: Confidence(1) + 5 Emotions = 6
        self._basic_features_count = 6

    def predict(self, input_vector_6d: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """
        Args:
            input_vector_6d: np.array([confidence, angry, fear, happy, neutral, sad])
        """
        # 1. Feature Engineering (Must match the logic used during Training!)
        full_feature_vector = self._create_advanced_features(input_vector_6d)

        # 2. Similarity Search
        return self.analyzer.find_most_similar_preset(full_feature_vector)

    def _create_advanced_features(self, vector: np.ndarray) -> np.ndarray:
        """
        Re-implements the logic from 'analyzer.create_advanced_features' using pure Numpy.
        Original columns: [conf, angry, fear, happy, neutral, sad]
        """
        confidence = vector[0]
        # Basic emotions slices
        probs = vector[1:]  # [angry, fear, happy, neutral, sad]

        p_angry = probs[0]
        p_fear = probs[1]
        p_happy = probs[2]
        p_neutral = probs[3]
        p_sad = probs[4]

        # --- Calculated Features ---

        # 1. Intensity (Max probability)
        intensity = np.max(probs)

        # 2. Entropy
        # Filter > 0 to avoid log(0).
        # Matches pandas logic: 0 * log(eps) -> 0, so we just skip zeros.
        valid_probs = probs[probs > 0]
        entropy = -np.sum(valid_probs * np.log(valid_probs + EPSILON))

        # 3. Valence Features
        neg_sum = p_angry + p_fear + p_sad
        pos_sum = p_happy + p_neutral
        balance = pos_sum - neg_sum

        # 4. Dominance Ratio
        sorted_p = np.sort(probs)
        dom_ratio = sorted_p[-1] / (sorted_p[-2] + EPSILON)

        conf_emo_ratio = confidence / (intensity + EPSILON)

        # 5. Distribution
        variance = np.var(probs, ddof=1)
        rng = np.max(probs) - np.min(probs)  # range

        # Concatenate all
        advanced = np.array([intensity, entropy, neg_sum, pos_sum, balance, dom_ratio, conf_emo_ratio, variance, rng])

        return np.concatenate([vector, advanced])
