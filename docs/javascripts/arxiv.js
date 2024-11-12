// 假设这是从某个地方获取的 arXiv 搜索结果，通常你会通过 AJAX 请求或者 fetch API 获取这个数据
const arxivResults = [
    {
        title: 'Example Paper Title 1',
        authors: ['Author1', 'Author2'],
        abstract: 'This is the abstract of the first example paper.',
        link: 'https://arxiv.org/abs/example1'
    },
    {
        title: 'Example Paper Title 2',
        authors: ['Author3', 'Author4'],
        abstract: 'This is the abstract of the second example paper.',
        link: 'https://arxiv.org/abs/example2'
    }
    // ... more results
];
 
// 获取页面上的某个元素来展示搜索结果
const resultsDiv = document.getElementById('arxiv-papers');
 
// 遍历搜索结果，并为每个结果创建一个展示元素
arxivResults.forEach(result => {
    const articleDiv = document.createElement('div');
    articleDiv.classList.add('article');
 
    const title = document.createElement('h3');
    title.textContent = result.title;
    articleDiv.appendChild(title);
 
    const authors = document.createElement('p');
    authors.textContent = `Authors: ${result.authors.join(', ')}`;
    articleDiv.appendChild(authors);
 
    const abstract = document.createElement('p');
    abstract.textContent = `Abstract: ${result.abstract}`;
    articleDiv.appendChild(abstract);
 
    const link = document.createElement('a');
    link.href = result.link;
    link.textContent = 'View on arXiv';
    articleDiv.appendChild(link);
 
    // 将创建好的元素添加到页面上
    resultsDiv.appendChild(articleDiv);
});