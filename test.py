import openai

# 🔹 Replace with your actual OpenAI API key
api_key = "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"

def test_openai_api():
    try:
        openai.api_key = api_key

        # 🔹 Send a simple test prompt
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Use "gpt-3.5-turbo" for cheaper tests
            messages=[{"role": "user", "content": "Hello, can you introduce yourself?"}],
            max_tokens=100
        )

        # 🔹 Print the response
        print("✅ OpenAI API Test Successful!")
        print("🔹 Model Response:", response["choices"][0]["message"]["content"])

    except openai.error.AuthenticationError:
        print("❌ ERROR: Invalid API key. Please check your OpenAI API key.")
    except openai.error.RateLimitError:
        print("⚠️ ERROR: Rate limit exceeded. Try again later or increase your quota.")
    except openai.error.APIConnectionError:
        print("🔌 ERROR: Connection issue. Check your internet or OpenAI server status.")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

# Run the test
test_openai_api()
