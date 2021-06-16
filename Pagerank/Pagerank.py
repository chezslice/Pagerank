import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Output link variable.
    if corpus[page]:
        # Calucate formula the random probability for the pages.
        random_factor = [(1 - damping_factor) / len(corpus)] * len(corpus)
        # Calucate formula the specific page-related probability.
        specific_factor = dict(zip(corpus.keys(), random_factor))

        # All pages linked by the current page, add additional probability.
        links = damping_factor / len(corpus[page])

        # For loop, to internate over corpus_links in the corpus. Include specific_factor and additional link probability.
        for corpus_links in corpus[page]:
            specific_factor[corpus_links] += links
        return specific_factor

    # No remaining links, probability chooses randomly over all pages that are equal.
    else:
        return dict(zip(corpus.keys(), [1 / len(corpus)] * len(corpus)))


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Intialize dictionary, set all values to 0.
    sample_ranks = 0
    # While sample_rank is 0, include dictionary formula.
    sample_ranks = dict(zip(corpus.keys(), [0] * len(corpus)))

    # Random page inside the corpus.
    corpus_page = random.choice(list(corpus.keys()))

    # For loop, to internate n total amount of times including the random variable.
    # Each sample increase counter for the current page, then get to next page that is based on the transition model function.
    for i in (range(n - 1)):
        sample_ranks[corpus_page] += 1
        internation = transition_model(corpus, corpus_page, damping_factor)
        corpus_page = random.choices(list(internation.keys()), internation.values())[0]

    # By using division, divide the page counts by n the answer will equal the proportion of sample for the page.
    sample_ranks = {corpus_page: num_samples/n for corpus_page, num_samples in sample_ranks.items()}

    return sample_ranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Create a variable total and set the variable to eqaul the length of the corpus.
    total = len(corpus)

    # Additional variables to the dictionary including variable formulas for each.
    interate_rank = dict(zip(corpus.keys(), [1/total] * total))
    interate_changes = dict(zip(corpus.keys(), [math.inf] * total))

    # Updating pageranks until values equal to 0.001 in interations.
    while any(interate_changes > 0.001 for interate_changes in interate_changes.values()):
        for page in interate_rank.keys():
            # Create a new link counter.
            link_counter = 0

            # For loop, to interate over page_links and links from the corpus.
            for page_link, links in corpus.items():
                # If no links are found, then one link is added.
                if not links:
                    links = corpus.keys()

                # If the page does have link increase link_counter from links.
                if page in links:
                    link_counter += interate_rank[page_link] / len(links)

            # Creating a new variable new_rank and intergrading the formula with the new variable.
            new_rank = ((1 - damping_factor) / total) + (damping_factor * link_counter)

            # Change between the old and new page_ranks.
            interate_changes[page] = abs(new_rank - interate_rank[page])
            interate_rank[page] = new_rank

    return interate_rank

#print("Code Completed")

if __name__ == "__main__":
    main()
