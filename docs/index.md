<html>
  <head>
    <title>arXiv API Example</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
    <script src="./javascripts/arxiv.js"></script>
    <style>
      .search-box {
        width: 300px;
        padding: 5px;
        background-color: #ffffff;
        border: 1px solid #000000;
        color: #333333;
        font-size: 16px;
      }
      /* 添加新样式以设置Title的颜色为蓝色 */
      .title {
        color: blue;
      }
    </style>
    <script> 
      function arxiv_html_search(title) {
        $.when(arxiv_search({ title: title })).then(function(data) {
          $('.results').empty();
          var promises = data.map(function(article, index) {
            return getPublicationRank("Computer Methods in Applied Mechanics and Engineering")
              .then(function(easyscholar_dict) {
                // 构建每个文章的 HTML，包括 Tag 信息
                var resultHTML = "<br />Title: <a href='" + article.link + "' class='title-link'>" + article.title + "</a><br />";
                //resultHTML += "Tag: SCI分区 " + easyscholar_dict.data.officialRank.all.sci + "  EI检索 " + easyscholar_dict.data.officialRank.all.eii + "  IF因子" + easyscholar_dict.data.officialRank.all.sciif + "  " + easyscholar_dict.data.officialRank.all.sciUpTop + "<br />";
                resultHTML += `
                  Tag: 
                    ${getStyledSpan("SCI分区 " + easyscholar_dict.data.officialRank.all.sci, 'rgb(74, 185, 72)')}  
                    ${getStyledSpan(easyscholar_dict.data.officialRank.all.eii, 'rgb(185, 134, 72)')}  
                    ${getStyledSpan("IF因子 " + easyscholar_dict.data.officialRank.all.sciif, 'rgb(74, 74, 185)')}  
                    ${getStyledSpan(easyscholar_dict.data.officialRank.all.sciUpTop, 'rgb(128, 128, 255)')}
                  <br />
                `;
                resultHTML += "Author: " + article.authors.join(', ') + "<br />";
                resultHTML += "Abstract: " + article.summary + "<br /><br />";
                return resultHTML;
              })
              .catch(function(error) {
                console.error('An error occurred while fetching publication rank:', error);
                // 返回一个占位符或错误信息，以便在 UI 中显示
                return "<br />Title: <a href='" + article.link + "' class='title-link'>" + article.title + "</a><br />";
                resultHTML += "Tag: Error fetching SCI rank<br />";
                resultHTML += "Author: " + article.authors.join(', ') + "<br />";
                resultHTML += "Abstract: " + article.summary + "<br /><br />";
              });
          });
      
          // 等待所有 Promise 解决，然后更新 HTML
          Promise.all(promises).then(function(results) {
            results.forEach(function(resultHTML) {
              $('.results').append(resultHTML);
            });
          });
        });
      }
    </script>
  </head>

  <body>
    <form>
      Title
      <br />
      <input type='text' name='doc_title' class='search-box' autofocus><br />
      <br />
      <input type="button" value="Search" onclick="arxiv_html_search(document.getElementsByName('doc_title')[0].value)">
    </form>
    <div class='results'>
    </div>
  </body>
</html>