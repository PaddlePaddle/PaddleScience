<html>
  <head>
    <title>arXiv API Example</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
    <script src="./javascripts/arxiv.js"></script>
    <script> 
      function arxiv_html_search(title) {
        $.when(arxiv_search({title: title})).then( function(data) { 
          $('.results').empty();
          for (var i = 0; i < data.length; ++i) {
            $('.results').append("Title: " + data[i].title + "<br />Author: " + data[i].authors[0] + "<br />URL: <a href='" + data[i].link + "'>" + data[i].link + "</a><br /><br />");
          }
        });
      }
    </script>
  </head>

  <body>
    <form>
      Title
      <br />
      <input type='text' name='doc_title'>
      <br />
      <input type="button" value="Search" onclick="arxiv_html_search(document.getElementsByName('doc_title')[0].value)">
    </form>
    <div class='results'>
    </div>
  </body>
</html>