import streamlit as st
from NutritionBot import NutritionBot

# Streamlit UI
def main():
    st.set_page_config(page_title="Nutrition Facts ChatBot", page_icon="🥦")
    
    # Initialize session state for chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize NutritionBot
    nutrition_bot = NutritionBot()
    
    # App title and description
    st.title("🥦 Nutrition Facts ChatBot")
    st.write("Ask questions about nutrition and health based on Dr. Michael Greger's Nutrition Facts videos!")
    
    # Predefined example questions
    example_questions = ["diverticulosis", "heart disease", "low carb diets", "diabetes", "green tea"]
    
    # Sidebar for example questions
    with st.sidebar:
        st.header("Example Questions")
        for example in example_questions:
            if st.button(example):
                st.session_state.chat_history.append([example, ""])
                response = nutrition_bot.process_chat(example, st.session_state.chat_history[:-1])
                st.session_state.chat_history[-1][1] = response
    
    # Display chat history
    for human, ai in st.session_state.chat_history:
        st.chat_message("human").write(human)
        st.chat_message("assistant").write(ai)
    
    # Chat input
    if prompt := st.chat_input("Ask me a question about nutrition and health"):
        st.session_state.chat_history.append([prompt, ""])
        with st.chat_message("human"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = nutrition_bot.process_chat(prompt, st.session_state.chat_history[:-1])
                st.write(response)
        
        st.session_state.chat_history[-1][1] = response

if __name__ == "__main__":
    main()