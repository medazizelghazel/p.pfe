from app.services.analysis_service import AnalysisService

service = AnalysisService()

result = service.run("data/input_videos/test7.mp4")

print("\nTRAINER EMOTION")
print(result.trainer_emotion)

print("\nLEARNER EMOTION ENABLED")
print(result.learner_emotion_enabled)

print("\nGLOBAL LEARNERS EMOTION")
print(result.learner_emotion)

print("\nLEARNER AUDIO")
print(result.learner_emotion_audio_path)
