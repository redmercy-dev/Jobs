import base64
import json
import traceback
import requests
from bs4 import BeautifulSoup
import streamlit as st
import os
import time
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# Note: Removed asyncio and nest_asyncio usage for simplicity and stability.
# Calling Streamlit commands from multiple threads or async contexts can cause "missing ScriptRunContext" errors.
# This version runs the calls synchronously and ensures Streamlit functions are only used in the main thread.

def extract_text_and_links_from_url(target_url):
    """Fetches the job posting links and their corresponding descriptions from the given URL."""
    PROXY_URL = 'https://proxy.scrapeops.io/v1/'
    API_KEY = 'c54dc5f9-4fbf-4516-9192-443cf13235e2'
    max_attempts = 2
    retry_delay = 1  # Initial retry delay in seconds

    def fetch_content_via_proxy(url):
        params = {
            'api_key': API_KEY,
            'url': url,
            'render_js': 'false',
            'residential': 'true',
        }
        for attempt in range(max_attempts):
            response = requests.get(PROXY_URL, params=params)
            if response.status_code == 200:
                return BeautifulSoup(response.text, 'html.parser')
            elif response.status_code == 500 and attempt == 0:
                print("Received status code 500, retrying...")
            elif response.status_code == 429:
                wait = retry_delay * (2 ** attempt)  # Exponential backoff for rate limiting
                print(f"Rate limit exceeded, retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"Failed to fetch the page, status code: {response.status_code}")
                return None
        print("All retries failed.")
        return None

    def scrape_job_description(full_url):
        """Fetches and extracts the job description text from the given job link."""
        soup = fetch_content_via_proxy(full_url)
        if soup:
            [script_or_style.decompose() for script_or_style in soup(['script', 'style'])]
            return ' '.join(soup.stripped_strings)
        return "Failed to fetch the job description."

    job_patterns = ['/vacancies/', '/job/']  # Patterns to identify job postings

    soup = fetch_content_via_proxy(target_url)
    if not soup:
        return "Failed to fetch the content."

    base_url = '{uri.scheme}://{uri.netloc}'.format(uri=requests.utils.urlparse(target_url))

    results = [
        {'text': ' '.join(link.stripped_strings).strip(), 'href': urljoin(base_url, link.get('href'))}
        for link in soup.find_all('a', href=True)
        if any(pattern in link.get('href') for pattern in job_patterns)
        and len(' '.join(link.stripped_strings).strip()) > 30
        and not ' '.join(link.stripped_strings).strip().startswith(('Copyright', '{'))
    ]

    def fetch_and_store_description(job_link):
        job_description = scrape_job_description(job_link['href'])
        return job_link['href'], job_description

    # Use ThreadPoolExecutor to fetch descriptions in parallel for the first 2 results
    max_workers = min(3, len(results)+1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_store_description, job_link): job_link for job_link in results[:2]}
        job_descriptions = {future.result()[0]: future.result()[1] for future in as_completed(futures)}

    # Update results with job descriptions
    for job_link in results:
        job_link['job_description'] = job_descriptions.get(job_link['href'], 'No description available')

    return results


def fetch_and_process_content(target_url):
    """Wrapper function to call the extract_text_and_links_from_url and return its results."""
    return extract_text_and_links_from_url(target_url)


# JSON definition for the new scraping tool
scrape_links_json = {
    "type": "function",
    "function": {
        "name": "fetch_and_process_content",
        "description": "Fetches content from a specified URL using a proxy, extracts links and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_url": {
                    "type": "string",
                    "description": "The URL of the webpage from which content is to be fetched."
                }
            },
            "required": ["target_url"]
        }
    }
}

tools = [
    scrape_links_json,
    {"type": "code_interpreter"},
    {"type": "file_search"}
]

available_functions = {
    "fetch_and_process_content": fetch_and_process_content,
}

instructions = """
You are a Job Description Analyzer that helps users discover suitable job opportunities directly from live web scraping. Instead of relying on uploaded job descriptions, you will use custom functions to scrape job postings from three specific websites:
- https://treuhand-job.ch
- https://www.jobagent.ch
- https://www.jobs.ch

Your goal is to recommend jobs and provide detailed information based on the user’s preferences and queries.

Steps to Follow:

Introduction:
1. Greet the user and introduce yourself as a Job Description Analyzer.
2. Explain that you will help them find the best job match by scraping live job postings.
3. Mention that you are equipped to gather results from the specified websites based on their needs.

Ask Initial Questions:
1. Inquire about the user’s preferences for the job they’re seeking, such as:
   - Desired job type (full-time, part-time, remote, on-site).
   - Industry or field of interest.
   - Specific skills or qualifications to be utilized.
   - Preferred job location.
   - Preferred companies or types of companies.
2. Confirm any additional criteria the user may have.

Scrape Job Descriptions:
1. Use the custom scraping function to gather job postings from the three specified websites.
2. Collect up to 3 relevant results at a time based on the user’s criteria.
3. If needed, adjust the custom function to get additional results or refine the queries.
4. When the user asks for jobs posted today, first use a code interpreter tool to determine today’s date and then specifically target the https://www.jobs.ch website for postings from that exact date.

Recommend Jobs:
1. Based on the user’s stated preferences, select and recommend the most suitable jobs from the scraped results.
2. Provide a brief overview of each recommended job:
   - Job title
   - Company name
   - Location
   - Key responsibilities or requirements
   - Contact information (email, phone) if publicly available
   - Salary range, if available
   - Direct links to the job postings to apply or learn more
3. Be concise in initial recommendations. If the user is interested in more details about a specific position, provide a more thorough description upon request.

Answer Questions:
1. Encourage the user to ask questions about any recommended job or to refine their preferences.
2. Provide detailed, helpful, and accurate answers based on newly scraped information if needed.
3. Offer additional options if the user’s preferences change or if they request more specific details.

Be Interactive:
1. Continuously refine recommendations based on feedback from the user.
2. Keep the conversation engaging by asking follow-up questions to clarify their interests and criteria.
3. Limit each scraping action to return only a few (up to 3) results at a time to maintain relevancy and prompt user feedback.

Additional Notes:
1. Always rely on the live scraping custom function for the latest job postings—no file upload is involved.
2. The user’s queries guide you in adjusting scraping parameters (e.g., filtering by industry, role, location).
3. For fresh “today’s postings,” verify today’s date and target https://www.jobs.ch to ensure accuracy.
4. Always provide public contact information if available.
5. The ultimate goal is to make the process of discovering job opportunities dynamic, relevant, and user-focused.
"""

def create_assistant(client, file_ids):
    assistant = client.beta.assistants.create(
        name="Jobs assistant",
        instructions=instructions,
        model="gpt-4o-mini",
        tools=tools,
        tool_resources={
            'file_search': {
                'vector_stores': [{
                    'file_ids': file_ids
                }]
            }
        }
    )
    return assistant.id


def safe_tool_call(func, tool_name, **kwargs):
    """Safely execute a tool call and handle exceptions."""
    try:
        result = func(**kwargs)
        return result if result is not None else f"No content returned from {tool_name}"
    except Exception as e:
        st.error(f"Error in {tool_name}: {str(e)}")
        return f"Error occurred in {tool_name}: {str(e)}"


def handle_tool_outputs(client, run, thread_id):
    tool_outputs = []
    try:
        if not run.required_action or not run.required_action.submit_tool_outputs:
            return run
        for call in run.required_action.submit_tool_outputs.tool_calls:
            function_name = call.function.name
            function = available_functions.get(function_name)
            if not function:
                raise ValueError(f"Function {function_name} not found in available_functions.")
            arguments = json.loads(call.function.arguments)
            with st.spinner(f"Executing a detailed search..."):
                output = safe_tool_call(function, function_name, **arguments)

            tool_outputs.append({
                "tool_call_id": call.id,
                "output": json.dumps(output)
            })

        return client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    except Exception as e:
        st.error(f"Error in handle_tool_outputs: {str(e)}")
        st.error(traceback.format_exc())
        return run


def get_agent_response(client, assistant_id, user_message, thread_id):
    with st.spinner("Processing your request..."):
        # Post user message
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message,
        )

        # Create a run
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        # Poll run status
        while run.status in ["queued", "in_progress", "requires_action"]:
            if run.status == "requires_action":
                # Handle tool outputs if required
                run = handle_tool_outputs(client, run, thread_id)
            else:
                time.sleep(1)

            if run.status not in ["queued", "in_progress", "requires_action"]:
                break

            # Update run status
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )

        # Fetch the last assistant message
        messages = client.beta.threads.messages.list(thread_id=thread_id, limit=1).data
        if not messages:
            return "Error: No assistant response", [], []

        last_message = messages[0]

        formatted_response_text = ""
        download_links = []
        images = []

        if last_message.role == "assistant":
            for content in last_message.content:
                if content.type == "text":
                    formatted_response_text += content.text.value
                    for annotation in content.text.annotations:
                        if annotation.type == "file_path":
                            file_id = annotation.file_path.file_id
                            file_name = annotation.text.split('/')[-1]
                            file_content = client.files.content(file_id).read()
                            download_links.append((file_name, file_content))
                elif content.type == "image_file":
                    file_id = content.image_file.file_id
                    image_data = client.files.content(file_id).read()
                    images.append((f"{file_id}.png", image_data))
                    formatted_response_text += f"[Image generated: {file_id}.png]\n"
        else:
            formatted_response_text = "Error: No assistant response"

        return formatted_response_text, download_links, images


def main():
    st.title("Jobs assistant")

    # Retrieve API keys from secrets inside main thread
    openai_api_key = st.secrets["api_keys"]["openai_api_key"]
    proxy_api_key = st.secrets["api_keys"]["proxy_api_key"]

    # Import and initialize the OpenAI client with the secure API key
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    # Global thread initialization
    if 'user_thread' not in st.session_state:
        st.session_state.user_thread = client.beta.threads.create()

    # Sidebar for assistant selection
    st.sidebar.title("Assistant Configuration")
    assistant_choice = st.sidebar.radio("Choose an option:", ["Create New Assistant", "Use Existing Assistant"])

    if assistant_choice == "Create New Assistant":
        uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
        file_ids = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_info = client.files.create(file=uploaded_file, purpose='assistants')
                file_ids.append(file_info.id)

        if file_ids:
            if st.sidebar.button("Create New Assistant"):
                st.session_state.assistant_id = create_assistant(client, file_ids)
                st.sidebar.success(f"New assistant created with ID: {st.session_state.assistant_id}")
        else:
            st.sidebar.warning("Please upload files to create an assistant.")

    else:  # Use Existing Assistant
        assistant_id = st.sidebar.text_input("Enter existing assistant ID:")
        if assistant_id:
            st.session_state.assistant_id = assistant_id
            st.sidebar.success(f"Using assistant with ID: {assistant_id}")

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "downloads" in message:
                for file_name, file_content in message["downloads"]:
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_content,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
                    if file_name.endswith('.html'):
                        st.components.v1.html(file_content.decode(), height=300, scrolling=True)
            if "images" in message:
                for image_name, image_data in message["images"]:
                    st.image(image_data)
                    st.download_button(
                        label=f"Download {image_name}",
                        data=image_data,
                        file_name=image_name,
                        mime="image/png"
                    )

    prompt = st.chat_input("You:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if 'assistant_id' in st.session_state:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response, download_links, images = get_agent_response(client, st.session_state.assistant_id, prompt, st.session_state.user_thread.id)
                message_placeholder.markdown(response)

                for file_name, file_content in download_links:
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_content,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
                    if file_name.endswith('.html'):
                        st.components.v1.html(file_content.decode(), height=300, scrolling=True)

                for image_name, image_data in images:
                    st.image(image_data)
                    st.download_button(
                        label=f"Download {image_name}",
                        data=image_data,
                        file_name=image_name,
                        mime="image/png"
                    )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "downloads": download_links,
                "images": images
            })
        else:
            st.warning("Please create a new assistant or enter an existing assistant ID before chatting.")


if __name__ == "__main__":
    main()
