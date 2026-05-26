from app.domain.analysis_result import AnalysisResult
from app.services.audio_extractor import AudioExtractor
from app.services.audio_preprocessor import AudioPreprocessor
from app.services.feature_extractor import FeatureExtractor
from app.services.clarity_evaluator import ClarityEvaluator
from app.services.engagement_estimator import EngagementEstimator
from app.services.score_aggregator import ScoreAggregator
from app.services.report_generator import ReportGenerator
from app.services.emotion_classifier import EmotionClassifier
from app.services.result_exporter import ResultExporter
from app.services.trainer_audio_builder import TrainerAudioBuilder
from app.services.learner_audio_builder import LearnerAudioBuilder
from app.utils.file_utils import build_analysis_id


class AnalysisService:
    def __init__(self):
        self.audio_extractor = AudioExtractor()
        self.audio_preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.clarity_evaluator = ClarityEvaluator()
        self.engagement_estimator = EngagementEstimator()
        self.score_aggregator = ScoreAggregator()
        self.report_generator = ReportGenerator()
        self.emotion_classifier = EmotionClassifier()
        self.result_exporter = ResultExporter()
        self.trainer_audio_builder = TrainerAudioBuilder()
        self.learner_audio_builder = LearnerAudioBuilder()

        self.diarization_service = None
        self.diarization_postprocessor = None

        self.transcription_service = None
        self.summary_generator = None

        try:
            from app.services.diarization_service import DiarizationService
            from app.services.diarization_postprocessor import DiarizationPostProcessor

            self.diarization_service = DiarizationService()
            self.diarization_postprocessor = DiarizationPostProcessor()
            print("[AnalysisService] Diarization enabled.")

        except Exception as e:
            print(f"[AnalysisService] Diarization disabled: {e}")

        try:
            from app.services.transcription_service import TranscriptionService
            from app.services.summary_generator import SummaryGenerator

            self.transcription_service = TranscriptionService(
                model_size="small",
                language=None,
                min_segment_duration=1.2,
            )
            self.summary_generator = SummaryGenerator(provider="mistral")
            print("[AnalysisService] Transcription and summary enabled.")

        except Exception as e:
            print(f"[AnalysisService] Transcription/Summary disabled: {e}")
            self.transcription_service = None
            self.summary_generator = None

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_section(self, title: str) -> None:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

    def _print_clarity_summary(self, clarity_result: dict) -> None:
        self._print_section("CLARITY ANALYSIS")

        if "num_chunks" in clarity_result:
            print(f"Chunks used                 : {clarity_result['num_chunks']}")

        print(f"Raw predicted score          : {clarity_result['raw_predicted_score']:.4f}")
        print(f"Clarity score /10            : {clarity_result['clarity_score_10']:.4f}")
        print(f"Clarity score /100           : {clarity_result['clarity_score']:.2f}")
        print(f"Clarity label                : {clarity_result['clarity_label']}")

    def _print_emotion_summary(self, emotion_result: dict) -> None:
        self._print_section("TRAINER EMOTION ANALYSIS")

        print(f"Dominant emotion             : {emotion_result['dominant_emotion']}")
        print(f"Confidence                   : {emotion_result['confidence']:.4f}")
        print(f"Analyzed segments            : {emotion_result['num_segments']}")
        print(f"Model used                   : {emotion_result['model_name']}")
        print(f"Aggregated scores            : {emotion_result.get('aggregated_scores', {})}")

    def _print_learner_emotion_summary(self, learner_emotion_result: dict | None) -> None:
        self._print_section("GLOBAL LEARNERS EMOTION ANALYSIS")

        if not learner_emotion_result or not learner_emotion_result.get("enabled"):
            print("Learner emotion enabled      : False")
            if learner_emotion_result:
                print(f"Reason/Error                 : {learner_emotion_result.get('reason') or learner_emotion_result.get('error')}")
            return

        print("Learner emotion enabled      : True")
        print(f"Type                         : {learner_emotion_result.get('type')}")
        print(f"Dominant learners emotion    : {learner_emotion_result.get('dominant_emotion')}")
        print(f"Confidence                   : {learner_emotion_result.get('confidence'):.4f}")
        print(f"Analyzed emotion segments    : {learner_emotion_result.get('num_segments')}")
        print(f"Learner audio duration       : {learner_emotion_result.get('audio_duration')} s")
        print(f"Learner diarization segments : {learner_emotion_result.get('learner_segment_count')}")
        print(f"Learner speakers             : {learner_emotion_result.get('learner_speaker_count')}")
        print(f"Aggregated scores            : {learner_emotion_result.get('aggregated_scores', {})}")

    def _print_engagement_summary(self, engagement_result: dict) -> None:
        details = engagement_result.get("details", {})

        self._print_section("ENGAGEMENT ANALYSIS")

        print(f"Engagement score /100        : {engagement_result['engagement_score']:.2f}")
        print(f"Engagement label             : {engagement_result['engagement_label']}")
        print(f"Engagement method            : {engagement_result['method']}")
        print(f"Uses diarization context     : {details.get('has_diarization_context', False)}")

        print("\n--- Components ---")
        print(f"Trainer vocal component      : {details.get('trainer_vocal_component', 0.0):.2f}/100")
        print(f"Learner participation        : {details.get('learner_participation_component', 0.0):.2f}/100")
        print(f"Trainer/Learner interaction  : {details.get('interaction_component', 0.0):.2f}/100")

        print("\n--- Trainer vocal indicators ---")
        print(f"Pitch std                    : {details.get('pitch_std', 0.0)}")
        print(f"Energy std                   : {details.get('energy_std', 0.0)}")
        print(f"Voiced ratio                 : {details.get('voiced_ratio', 0.0)}")
        print(f"Speech ratio                 : {details.get('speech_ratio', 0.0)}")
        print(f"Pause ratio                  : {details.get('pause_ratio', 0.0)}")
        print(f"Onset rate / sec             : {details.get('onset_rate_per_sec', 0.0)}")

        print("\n--- Emotion indicators used for engagement ---")
        print(f"Neutral / calm               : {details.get('emo_neutral_calm', 0.0)}")
        print(f"Energetic / engaged          : {details.get('emo_energetic_engaged', 0.0)}")
        print(f"Low energy                   : {details.get('emo_low_energy', 0.0)}")
        print(f"Tense / stressed             : {details.get('emo_tense_stressed', 0.0)}")

        if details.get("has_diarization_context", False):
            print("\n--- Learner participation indicators ---")
            print(f"Session duration             : {details.get('session_duration_sec', 0.0)} s")
            print(f"Trainer speech duration      : {details.get('trainer_total_speech_duration', 0.0)} s")
            print(f"Learner speech duration      : {details.get('learner_total_speech_duration', 0.0)} s")
            print(f"Trainer talk ratio           : {details.get('trainer_talk_ratio', 0.0)}")
            print(f"Learner talk ratio           : {details.get('learner_talk_ratio', 0.0)}")
            print(f"Trainer speech share         : {details.get('trainer_speech_share', 0.0)}")
            print(f"Learner speech share         : {details.get('learner_speech_share', 0.0)}")

            print("\n--- Interaction indicators ---")
            print(f"Trainer turn count           : {details.get('trainer_turn_count', 0)}")
            print(f"Learner turn count           : {details.get('learner_turn_count', 0)}")
            print(f"Total turn count             : {details.get('total_turn_count', 0)}")
            print(f"Active learners detected     : {details.get('active_learner_speaker_count', 0)}")
            print(f"Learner avg turn duration    : {details.get('learner_avg_turn_duration', 0.0)} s")
            print(f"Learner max turn duration    : {details.get('learner_max_turn_duration', 0.0)} s")
            print(f"Learner turns / min          : {details.get('learner_turns_per_min', 0.0)}")
            print(f"Interaction turn count       : {details.get('interaction_turn_count', 0)}")
            print(f"Interaction rate / min       : {details.get('interaction_rate_per_min', 0.0)}")
            print(f"Trainer/Learner balance      : {details.get('trainer_learner_balance', 0.0)}")
            print(f"Learner response count       : {details.get('learner_response_count', 0)}")
            print(f"Learner response ratio       : {details.get('learner_response_ratio', 0.0)}")
            print(f"Overlap segment count        : {details.get('overlap_segment_count', 0)}")
            print(f"Overlap ratio                : {details.get('overlap_ratio', 0.0)}")

    def _print_transcription_summary(self, transcription_result: dict | None) -> None:
        self._print_section("TRANSCRIPTION ANALYSIS")

        if not transcription_result:
            print("Transcription enabled        : False")
            return

        print("Transcription enabled        : True")
        print(f"Trainer ID                   : {transcription_result.get('trainer_id')}")
        print(f"Course language              : {transcription_result.get('course_language')}")
        print(f"Language confidence          : {transcription_result.get('language_confidence')}")
        print(f"Segments transcribed         : {transcription_result.get('segments_count')}")
        print(f"Total transcript duration    : {transcription_result.get('total_duration')} s")
        print(f"Transcript JSON              : {transcription_result.get('output_json')}")
        print(f"Transcript TXT               : {transcription_result.get('output_txt')}")

    def _print_summary_result(self, summary_result: dict | None) -> None:
        self._print_section("COURSE SUMMARY")

        if not summary_result:
            print("Summary enabled              : False")
            return

        summary = summary_result.get("summary", {})
        sections = summary.get("sections", [])
        points = summary.get("points_importants", [])

        print("Summary enabled              : True")
        print(f"Course title                 : {summary.get('titre_cours')}")
        print(f"Course language              : {summary.get('langue')}")
        print(f"Sections count               : {len(sections)}")
        print(f"Key points count             : {len(points)}")
        print(f"Summary JSON                 : {summary_result.get('output_json')}")

    def _print_diarization_summary(
        self,
        diarization_raw,
        diarization_post,
        trainer_audio_result,
        learner_audio_result,
        scoring_audio_source: str,
    ) -> None:
        self._print_section("DIARIZATION / AUDIO SOURCE")

        enabled = diarization_raw is not None and diarization_post is not None
        print(f"Diarization enabled          : {enabled}")
        print(f"Scoring audio source         : {scoring_audio_source}")

        if diarization_post:
            print(f"Trainer speaker ID           : {diarization_post['trainer']}")
            print(f"Trainer confidence           : {diarization_post['confidence']}")
            print(f"Diarization clean CSV        : {diarization_post['clean_csv']}")
            print(f"Profiles JSON                : {diarization_post['profiles_json']}")

        if diarization_raw:
            print(f"Raw diarization CSV          : {diarization_raw['output_csv']}")
            print(f"RTTM path                    : {diarization_raw['output_rttm']}")
            print(f"Real speakers                : {diarization_raw['real_speakers']}")
            print(f"Artifact speakers            : {diarization_raw['artifact_speakers']}")

        if trainer_audio_result:
            print(f"Trainer-only audio           : {trainer_audio_result['output_audio_path']}")
            print(f"Trainer-only duration        : {trainer_audio_result['concatenated_duration']} s")
            print(f"Trainer-only segments        : {trainer_audio_result['segment_count']}")
            print(f"Trainer segments CSV         : {trainer_audio_result['output_segments_csv']}")

        if learner_audio_result:
            print(f"Learners-only enabled        : {learner_audio_result.get('enabled')}")
            print(f"Learners-only audio          : {learner_audio_result.get('audio_path')}")
            print(f"Learners-only duration       : {learner_audio_result.get('duration')} s")
            print(f"Learner segments             : {learner_audio_result.get('segment_count')}")
            print(f"Learner speakers             : {learner_audio_result.get('speaker_count')}")
            print(f"Learner segments CSV         : {learner_audio_result.get('segments_csv_path')}")

    def _print_final_summary(
        self,
        video_path: str,
        processed_audio_path: str,
        processed_duration: float,
        clarity_score: float,
        engagement_score: float,
        global_score: float,
        dominant_emotion: str,
        emotion_confidence: float,
        learner_emotion_result: dict | None,
        report_path: str,
        json_path: str,
        transcription_result: dict | None,
        summary_result: dict | None,
    ) -> None:
        self._print_section("FINAL ANALYSIS SUMMARY")

        print(f"Video path                   : {video_path}")
        print(f"Processed audio              : {processed_audio_path}")
        print(f"Processed duration           : {processed_duration:.2f} s")
        print(f"Clarity score                : {clarity_score:.2f}/100")
        print(f"Engagement score             : {engagement_score:.2f}/100")
        print(f"Global score                 : {global_score:.2f}/100")
        print(f"Trainer dominant emotion     : {dominant_emotion}")
        print(f"Trainer emotion confidence   : {emotion_confidence:.4f}")

        if learner_emotion_result and learner_emotion_result.get("enabled"):
            print(f"Learners dominant emotion    : {learner_emotion_result.get('dominant_emotion')}")
            print(f"Learners emotion confidence  : {learner_emotion_result.get('confidence'):.4f}")
        else:
            print("Learners emotion             : not available")

        print(f"PDF report                   : {report_path}")
        print(f"JSON results                 : {json_path}")

        if transcription_result:
            print(f"Transcript JSON              : {transcription_result.get('output_json')}")

        if summary_result:
            print(f"Summary JSON                 : {summary_result.get('output_json')}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, video_path: str) -> AnalysisResult:
        analysis_id = build_analysis_id(video_path)

        extracted_audio_path = self.audio_extractor.extract(
            video_path,
            analysis_id,
        )

        diarization_raw = None
        diarization_post = None
        trainer_audio_result = None
        learner_audio_result = None

        trainer_emotion_result = None
        learner_emotion_result = None

        transcription_result = None
        summary_result = None
        summary_data = {}

        if self.diarization_service and self.diarization_postprocessor:
            diarization_output_dir = "data/results/diarization"

            diarization_raw = self.diarization_service.diarize_audio(
                audio_path=extracted_audio_path,
                output_dir=diarization_output_dir,
                min_speakers=2,
                max_speakers=3,
            )

            diarization_post = self.diarization_postprocessor.process(
                csv_path=diarization_raw["output_csv"],
                output_dir=diarization_output_dir,
                real_speakers=diarization_raw["real_speakers"],
            )

            try:
                trainer_audio_result = self.trainer_audio_builder.build(
                    audio_path=extracted_audio_path,
                    clean_csv_path=diarization_post["clean_csv"],
                    trainer_speaker_id=diarization_post["trainer"],
                    output_dir=diarization_output_dir,
                    analysis_id=analysis_id,
                )

                print(
                    "Trainer-only audio built: "
                    f"{trainer_audio_result['output_audio_path']}"
                )

            except Exception as e:
                print(
                    "Trainer-only audio build failed, "
                    f"fallback to full audio: {e}"
                )
                trainer_audio_result = None

            try:
                learner_audio_result = self.learner_audio_builder.build(
                    audio_path=extracted_audio_path,
                    clean_csv_path=diarization_post["clean_csv"],
                    output_dir=diarization_output_dir,
                    analysis_id=analysis_id,
                )

                if learner_audio_result.get("enabled"):
                    print(
                        "Learners-only audio built: "
                        f"{learner_audio_result['audio_path']}"
                    )
                else:
                    print(
                        "[AnalysisService] Learner audio disabled: "
                        f"{learner_audio_result.get('reason')}"
                    )

            except Exception as e:
                print(f"[AnalysisService] Learner audio build failed: {e}")
                learner_audio_result = None

            if (
                self.transcription_service
                and self.summary_generator
                and diarization_post
                and diarization_post.get("clean_csv")
            ):
                try:
                    transcription_result = self.transcription_service.transcribe_trainer(
                        audio_path=extracted_audio_path,
                        clean_csv=diarization_post["clean_csv"],
                        output_dir="data/results/transcription",
                    )

                    summary_result = self.summary_generator.generate(
                        transcript_json=transcription_result["output_json"],
                        output_dir="data/results/summary",
                    )

                    summary_data = summary_result.get("summary", {})

                except Exception as e:
                    print(f"[AnalysisService] Transcription/Summary failed: {e}")
                    transcription_result = None
                    summary_result = None
                    summary_data = {}

        scoring_audio_input = extracted_audio_path
        scoring_audio_source = "full_audio"

        if trainer_audio_result is not None:
            scoring_audio_input = trainer_audio_result["output_audio_path"]
            scoring_audio_source = "trainer_only"

        processed_audio = self.audio_preprocessor.process(
            scoring_audio_input,
            analysis_id,
        )

        features = self.feature_extractor.extract(processed_audio.path)

        if trainer_audio_result and trainer_audio_result.get("chunk_count", 0) > 0:
            clarity_result = self.clarity_evaluator.evaluate_chunks(
                trainer_audio_result["output_chunks_dir"]
            )
        else:
            clarity_result = self.clarity_evaluator.evaluate(processed_audio.path)

        clarity_score = clarity_result["clarity_score"]
        self._print_clarity_summary(clarity_result)

        # Trainer emotion is computed on processed_audio.path.
        # If diarization succeeds, processed_audio.path is based on trainer-only audio.
        emotion_result = self.emotion_classifier.classify_with_details(
            processed_audio.path
        )

        dominant_emotion = emotion_result["dominant_emotion"]
        emotion_scores = emotion_result.get("aggregated_scores", {})

        trainer_emotion_result = {
            "enabled": True,
            "type": "trainer",
            "dominant_emotion": emotion_result.get("dominant_emotion"),
            "confidence": emotion_result.get("confidence"),
            "num_segments": emotion_result.get("num_segments"),
            "aggregated_scores": emotion_result.get("aggregated_scores", {}),
            "model_name": emotion_result.get("model_name"),
            "model_metrics": emotion_result.get("model_metrics", {}),
        }

        self._print_emotion_summary(emotion_result)

        # Global emotion of all learners.
        # This uses all learner segments already identified in the clean diarization CSV.
        if learner_audio_result and learner_audio_result.get("enabled"):
            try:
                learner_emotion_raw = self.emotion_classifier.classify_with_details(
                    learner_audio_result["audio_path"]
                )

                learner_emotion_result = {
                    "enabled": True,
                    "type": "global_all_learners",
                    "dominant_emotion": learner_emotion_raw.get("dominant_emotion"),
                    "confidence": learner_emotion_raw.get("confidence"),
                    "num_segments": learner_emotion_raw.get("num_segments"),
                    "aggregated_scores": learner_emotion_raw.get("aggregated_scores", {}),
                    "model_name": learner_emotion_raw.get("model_name"),
                    "model_metrics": learner_emotion_raw.get("model_metrics", {}),
                    "audio_path": learner_audio_result["audio_path"],
                    "segments_csv_path": learner_audio_result["segments_csv_path"],
                    "audio_duration": learner_audio_result["duration"],
                    "learner_segment_count": learner_audio_result["segment_count"],
                    "learner_speaker_count": learner_audio_result["speaker_count"],
                    "learner_speakers": learner_audio_result["speakers"],
                }

            except Exception as e:
                print(f"[AnalysisService] Learner emotion analysis failed: {e}")
                learner_emotion_result = {
                    "enabled": False,
                    "type": "global_all_learners",
                    "error": str(e),
                }
        else:
            learner_emotion_result = {
                "enabled": False,
                "type": "global_all_learners",
                "reason": "No valid learner audio available.",
            }

        self._print_learner_emotion_summary(learner_emotion_result)

        diarization_clean_csv_path = (
            diarization_post["clean_csv"]
            if diarization_post
            else None
        )

        trainer_speaker_id = (
            diarization_post["trainer"]
            if diarization_post
            else None
        )

        engagement_result = self.engagement_estimator.estimate_from_audio(
            audio_path=processed_audio.path,
            emotion_scores=emotion_scores,
            diarization_csv_path=diarization_clean_csv_path,
            trainer_speaker_id=trainer_speaker_id,
        )

        engagement_score = engagement_result["engagement_score"]
        engagement_label = engagement_result["engagement_label"]
        engagement_method = engagement_result["method"]
        engagement_details = engagement_result["details"]

        self._print_engagement_summary(engagement_result)
        self._print_transcription_summary(transcription_result)
        self._print_summary_result(summary_result)

        global_score = self.score_aggregator.aggregate(
            clarity_score=clarity_score,
            engagement_score=engagement_score,
            trainer_emotion=trainer_emotion_result,
            learner_emotion=learner_emotion_result,
        )

        result = AnalysisResult(
            video_path=video_path,
            extracted_audio_path=extracted_audio_path,
            processed_audio_path=processed_audio.path,
            sample_rate=processed_audio.sample_rate,
            duration=processed_audio.duration,
            features=features,

            clarity_score=clarity_score,
            engagement_score=engagement_score,
            engagement_label=engagement_label,
            engagement_method=engagement_method,
            engagement_details=engagement_details,
            global_score=global_score,

            dominant_emotion=dominant_emotion,
            emotion_confidence=emotion_result["confidence"],
            emotion_num_segments=emotion_result["num_segments"],
            emotion_aggregated_scores=emotion_result["aggregated_scores"],
            emotion_segment_predictions=emotion_result["segment_predictions"],
            emotion_model_used=emotion_result["model_name"],
            emotion_model_metrics=emotion_result["model_metrics"],

            trainer_emotion=trainer_emotion_result,

            learner_emotion_enabled=bool(
                learner_emotion_result and learner_emotion_result.get("enabled")
            ),
            learner_emotion=learner_emotion_result,
            learner_emotion_audio_path=(
                learner_emotion_result.get("audio_path")
                if learner_emotion_result and learner_emotion_result.get("enabled")
                else None
            ),
            learner_emotion_segments_csv_path=(
                learner_emotion_result.get("segments_csv_path")
                if learner_emotion_result and learner_emotion_result.get("enabled")
                else None
            ),
            learner_emotion_segment_count=(
                learner_emotion_result.get("learner_segment_count", 0)
                if learner_emotion_result
                else 0
            ),
            learner_emotion_speaker_count=(
                learner_emotion_result.get("learner_speaker_count", 0)
                if learner_emotion_result
                else 0
            ),
            learner_emotion_audio_duration=(
                learner_emotion_result.get("audio_duration", 0.0)
                if learner_emotion_result
                else 0.0
            ),

            diarization_enabled=diarization_raw is not None and diarization_post is not None,
            diarization_rttm_path=diarization_raw["output_rttm"] if diarization_raw else None,
            diarization_raw_csv_path=diarization_raw["output_csv"] if diarization_raw else None,
            diarization_clean_csv_path=diarization_clean_csv_path,
            trainer_speaker_id=trainer_speaker_id,
            trainer_detection_confidence=diarization_post["confidence"] if diarization_post else None,
            diarization_real_speakers=diarization_raw["real_speakers"] if diarization_raw else [],
            diarization_artifact_speakers=diarization_raw["artifact_speakers"] if diarization_raw else [],
            diarization_profiles_json_path=diarization_post["profiles_json"] if diarization_post else None,

            scoring_audio_source=scoring_audio_source,
            trainer_audio_path=trainer_audio_result["output_audio_path"] if trainer_audio_result else None,
            trainer_audio_segments_csv_path=trainer_audio_result["output_segments_csv"] if trainer_audio_result else None,
            trainer_audio_segment_count=trainer_audio_result["segment_count"] if trainer_audio_result else 0,
            trainer_audio_duration=trainer_audio_result["concatenated_duration"] if trainer_audio_result else 0.0,

            transcription_enabled=transcription_result is not None,
            transcript_json_path=transcription_result["output_json"] if transcription_result else None,
            transcript_txt_path=transcription_result["output_txt"] if transcription_result else None,
            transcript_language=transcription_result["course_language"] if transcription_result else None,
            transcript_language_confidence=transcription_result["language_confidence"] if transcription_result else 0.0,
            transcript_segments_count=transcription_result["segments_count"] if transcription_result else 0,
            transcript_total_duration=transcription_result["total_duration"] if transcription_result else 0.0,

            summary_enabled=summary_result is not None,
            summary_json_path=summary_result["output_json"] if summary_result else None,
            course_title=summary_data.get("titre_cours"),
            course_language=summary_data.get("langue"),
            course_sections_count=len(summary_data.get("sections", [])),
            course_key_points=summary_data.get("points_importants", []),
            course_summary=summary_data,
        )

        report_path = self.report_generator.generate(result, analysis_id)
        json_path = self.result_exporter.export_json(result, analysis_id)

        self._print_diarization_summary(
            diarization_raw=diarization_raw,
            diarization_post=diarization_post,
            trainer_audio_result=trainer_audio_result,
            learner_audio_result=learner_audio_result,
            scoring_audio_source=scoring_audio_source,
        )

        self._print_final_summary(
            video_path=video_path,
            processed_audio_path=processed_audio.path,
            processed_duration=processed_audio.duration,
            clarity_score=clarity_score,
            engagement_score=engagement_score,
            global_score=global_score,
            dominant_emotion=dominant_emotion,
            emotion_confidence=emotion_result["confidence"],
            learner_emotion_result=learner_emotion_result,
            report_path=report_path,
            json_path=json_path,
            transcription_result=transcription_result,
            summary_result=summary_result,
        )

        return result
    