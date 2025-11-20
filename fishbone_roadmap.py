from typing import List, Dict, Any
import re
from inference import semantic_search


def is_youtube_video(link: str) -> bool:
    """Check if a link is a YouTube video"""
    if not link:
        return False
    youtube_patterns = [
        r'youtube\.com/watch',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'youtube\.com/v/'
    ]
    return any(re.search(pattern, link, re.IGNORECASE) for pattern in youtube_patterns)


def is_project_resource(item: Dict[str, Any]) -> bool:
    """Check if an item is a project resource (dataset, kaggle, github project, etc.)"""
    content_type = item.get("content_type", "").lower()
    source = item.get("source", "").lower()
    link = item.get("link", "").lower()
    title = item.get("title", "").lower()
    
    # Check content type
    if content_type in ["dataset", "project", "code"]:
        return True
    
    # Check source
    if source in ["kaggle", "github", "gitlab"]:
        return True
    
    # Check link patterns
    project_patterns = [
        r'kaggle\.com/(datasets|code|competitions)',
        r'github\.com/.*/(project|example|tutorial)',
        r'github\.com/.*/.*',
    ]
    if any(re.search(pattern, link, re.IGNORECASE) for pattern in project_patterns):
        return True
    
    # Check title keywords
    project_keywords = ["project", "dataset", "example", "tutorial", "implementation", "code"]
    if any(keyword in title for keyword in project_keywords):
        return True
    
    return False


def is_certification_resource(item: Dict[str, Any]) -> bool:
    """Check if an item is a certification resource (course, specialization, certificate, etc.)"""
    content_type = item.get("content_type", "").lower()
    source = item.get("source", "").lower()
    link = item.get("link", "").lower()
    title = item.get("title", "").lower()
    
    # Check content type
    if content_type in ["course", "certification", "specialization", "certificate"]:
        return True
    
    # Check source
    if source in ["coursera", "udemy", "edx", "udacity", "freecodecamp", "pluralsight"]:
        return True
    
    # Check link patterns
    cert_patterns = [
        r'coursera\.org/(learn|specializations|professional-certificates)',
        r'udemy\.com/course',
        r'edx\.org/course',
        r'freecodecamp\.org/learn',
    ]
    if any(re.search(pattern, link, re.IGNORECASE) for pattern in cert_patterns):
        return True
    
    # Check title keywords
    cert_keywords = ["certificate", "certification", "course", "specialization", "bootcamp", "professional certificate"]
    if any(keyword in title for keyword in cert_keywords):
        return True
    
    return False


def get_projects_and_certifications(query: str, top_k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    """Get projects and certifications specific to the query using semantic search"""
    # Search for resources related to the query
    search_results = semantic_search(query=query, top_k=top_k)
    
    projects = []
    certifications = []
    project_links = set()
    cert_links = set()
    
    for item in search_results:
        link = item.get("link", "")
        if is_project_resource(item) and link not in project_links:
            projects.append({
                "title": item.get("title", "Untitled"),
                "link": link,
                "platform": item.get("source", "Unknown"),
                "similarity_score": item.get("similarity_score", 0.0)
            })
            project_links.add(link)
        elif is_certification_resource(item) and link not in cert_links:
            certifications.append({
                "title": item.get("title", "Untitled"),
                "link": link,
                "platform": item.get("source", "Unknown"),
                "similarity_score": item.get("similarity_score", 0.0)
            })
            cert_links.add(link)
    
    # Sort by similarity score and limit to top results
    projects = sorted(projects, key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)[:4]
    certifications = sorted(certifications, key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)[:4]
    
    # If we don't have enough projects/certifications, search with more specific queries
    if len(projects) < 3:
        project_query = f"{query} project dataset implementation"
        project_results = semantic_search(query=project_query, top_k=15)
        for item in project_results:
            link = item.get("link", "")
            if is_project_resource(item) and link not in project_links:
                projects.append({
                    "title": item.get("title", "Untitled"),
                    "link": link,
                    "platform": item.get("source", "Unknown"),
                    "similarity_score": item.get("similarity_score", 0.0)
                })
                project_links.add(link)
        projects = sorted(projects, key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)[:4]
    
    if len(certifications) < 3:
        cert_query = f"{query} course certification specialization"
        cert_results = semantic_search(query=cert_query, top_k=15)
        for item in cert_results:
            link = item.get("link", "")
            if is_certification_resource(item) and link not in cert_links:
                certifications.append({
                    "title": item.get("title", "Untitled"),
                    "link": link,
                    "platform": item.get("source", "Unknown"),
                    "similarity_score": item.get("similarity_score", 0.0)
                })
                cert_links.add(link)
        certifications = sorted(certifications, key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)[:4]
    
    return {
        "projects": projects,
        "certifications": certifications
    }


def categorize_content(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize content into articles and videos"""
    articles = []
    videos = []
    
    for item in results:
        link = item.get("link", "")
        if is_youtube_video(link):
            videos.append(item)
        else:
            articles.append(item)
    
    return {
        "articles": articles,
        "videos": videos
    }


def build_fishbone_roadmap(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a fishbone structure roadmap with articles and videos separated"""
    
    # Categorize content
    categorized = categorize_content(results)
    
    # Sort by similarity score
    articles_sorted = sorted(
        categorized["articles"],
        key=lambda x: float(x.get("similarity_score", 0.0)),
        reverse=True
    )[:10]  # Top 10 articles
    
    videos_sorted = sorted(
        categorized["videos"],
        key=lambda x: float(x.get("similarity_score", 0.0)),
        reverse=True
    )[:10]  # Top 10 videos
    
    # Format articles
    formatted_articles = []
    for i, article in enumerate(articles_sorted, 1):
        formatted_articles.append({
            "id": i,
            "title": article.get("title", "Untitled"),
            "link": article.get("link", ""),
            "source": article.get("source", "Unknown"),
            "labels": article.get("labels", []),
            "credibility_score": article.get("credibility_score", 0.0),
            "similarity_score": article.get("similarity_score", 0.0)
        })
    
    # Format videos
    formatted_videos = []
    for i, video in enumerate(videos_sorted, 1):
        formatted_videos.append({
            "id": i,
            "title": video.get("title", "Untitled"),
            "link": video.get("link", ""),
            "source": video.get("source", "YouTube"),
            "labels": video.get("labels", []),
            "credibility_score": video.get("credibility_score", 0.0),
            "similarity_score": video.get("similarity_score", 0.0)
        })
    
    # Get projects and certifications specific to the query
    projects_and_certs = get_projects_and_certifications(query)
    
    return {
        "query": query,
        "articles": formatted_articles,
        "videos": formatted_videos,
        "total_articles": len(formatted_articles),
        "total_videos": len(formatted_videos),
        "projects": projects_and_certs["projects"],
        "certifications": projects_and_certs["certifications"]
    }
