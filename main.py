"""
Takes a search term, gets some wikipedia results, and then summarizes using
openai's gpt-3 api.

Author: Chris Albon
Created: 2023-01-19
"""

# Import libraries
import argparse
from typing import List
from typing import Dict
import requests
import openai
import bs4
import constants


def get_wikipedia_article_urls(
    search_term: str, search_key: str, search_id: str
) -> List[str]:
    """
    This function takes a search term, google search api key, and google search engine id and returns a list of wikipedia article urls
    
    Parameters
    ----------
    search_term : str
        The search term
    search_key : str
        The google search api key
    search_id : str
        The google search engine id
        
    Returns
    -------
    article_urls : list
        A list of wikipedia article urls
    """
    # using the first page
    page = 1
    # constructing the URL
    # doc: https://developers.google.com/custom-search/v1/using_rest
    # calculating start, (page=2) => (start=11), (page=3) => (start=21)
    start = (page - 1) * 10 + 1
    url = f"https://www.googleapis.com/customsearch/v1?key={search_key}&cx={search_id}&q={search_term}&start={start}"

    # make the API request
    data = requests.get(url, timeout=100).json()

    # get the result items
    search_items = data.get("items")

    article_urls = []

    # iterate over 10 results found
    for _, search_item in enumerate(search_items, start=1):
        # extract the page url
        link = search_item.get("link")
        article_urls.append(link)

    return article_urls


def get_wikipedia_text(url: str, paragraph_number: int = 3) -> str:
    """
    This function takes a wikipedia url and returns the text of the first 3 paragraphs

    Parameters
    ----------
    url : str
        The wikipedia url

    Returns
    -------
    paras : list
        A list of the first 3 paragraphs of text

    """

    page = requests.get(url, timeout=100)
    soup = bs4.BeautifulSoup(page.content, "html.parser")
    paras = []
    for paragraph in soup.find_all("p"):
        if len(paragraph.text) > 50:
            paras.append(str(paragraph.text).strip())

    text = " ".join(paras[0:paragraph_number])

    return text


def create_prompt(search_question, content, link):
    """
    This function takes a search question, the content of a wikipedia article, and the link to the wikipedia article and returns a prompt for openai's gpt-3 api
   
    Parameters
    ----------
    search_question : str
        The question to be answered
    content : str
        The content of the wikipedia article
    link : str
        The link to the wikipedia article
   
    Returns
    -------
    prompt : str
        The prompt for openai's gpt-3 api

    """

    prompt = (
        # fstring with a variable called content and a variable called search_question
        f"""
        Act as if no information exists in the universe other that what is in this text:
        `{content}`
        Answer the following question and add this link in brackets {link}:
        {search_question}
        If the question is not answered in the text, say that you don't know.
        """
    )

    return prompt


def create_response(prompt: str) -> Dict:
    """"
    This function takes a prompt and returns the response from openai's gpt-3 api

    Parameters
    ----------
    prompt : str
        The prompt for openai's gpt-3 api

    Returns
    -------
    response : str
        The response from openai's gpt-3 api

    """

    openai.api_key = constants.OPENAI_KEY
    answer = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=1,
        best_of=3,
    )

    return answer


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="The question from the user", type=str)
    parser.add_argument("--debug", help="Run in debug mode", action="store_true")

    args = parser.parse_args()

    SEARCH_QUESTION = args.question

    # Get the wikipedia article urls
    wikipedia_article_urls = get_wikipedia_article_urls(
        SEARCH_QUESTION,
        constants.GOOGLE_SEARCH_API_KEY,
        constants.GOOGLE_SEARCH_ENGINE_ID,
    )

    # Get the wikipedia text
    WIKIPEDIA_TEXT = get_wikipedia_text(wikipedia_article_urls[0])

    # Create the prompt
    prompt = create_prompt(SEARCH_QUESTION, WIKIPEDIA_TEXT, wikipedia_article_urls[0])

    # Create the response
    response = create_response(prompt)

    if args.debug:
        print("--------------------")
        print("Question: " + "\n\n" + SEARCH_QUESTION)
        print("--------------------")
        print("Corpus: " + "\n\n" + WIKIPEDIA_TEXT)
        print("--------------------")
        print("Response: ")

    print(response.choices[0]["text"])
