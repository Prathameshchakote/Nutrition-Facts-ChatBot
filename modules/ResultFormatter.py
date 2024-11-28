class ResultFormatter:
    @staticmethod
    def format_result(answer, context):
        unique_videos = set((doc.metadata['title'], doc.metadata['videoId']) for doc in context)        
        titles_with_links = [f"{title}: https://www.youtube.com/watch?v={video_id}" for title, video_id in unique_videos]
        titles_string = '\n'.join(titles_with_links)
        titles_formatted = f"Relevant Videos:\n{titles_string}"
        response = f"{answer}\n\n{titles_formatted}"
        return response