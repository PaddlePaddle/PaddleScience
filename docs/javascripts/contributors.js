const owner = 'PaddlePaddle';
const repo = 'PaddleScience';

fetch(`https://api.github.com/repos/${owner}/${repo}/contributors?per_page=65&page=1`, {
    headers: {
        "Accept": "application/vnd.github.v3+json"
    }
})
.then(response => {
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
})
.then(data => {
    const contributorsDiv = document.getElementById('contributors');
    data.forEach(contributor => {
        const a = document.createElement('a');
        a.href = `https://github.com/${contributor.login}`;
        a.innerHTML = `<img class="avatar" src="${contributor.avatar_url}" alt="avatar" />`;
        contributorsDiv.appendChild(a);
    });
})
.catch((error) => {
    console.error('Fetching contributors failed:', error);
});
