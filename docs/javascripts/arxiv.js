/* Copyright (c) 2016 Frase
 *
 * Distributed under MIT license (see LICENSE).
 *
 *
 * Search arXiv via its HTTP API
 *
 * can search the following 1 or more fields:
 *   - author
 *   - title
 *   - abstract
 *   - journal reference
 *   - All fields
 *   journal's referenced, as well as all fields.
 */



/**
 * Searches arXiv for papers/documents that match the supplied parameters
 * @param {string} all
 * @param {string} author
 * @param {string} title
 * @param {string} abstrct
 * @param {string} journal_ref
 * @returns {Promise}
 */
function arxiv_search({all, author, title, abstrct, journal_ref}) {
    var baseUrl = "https://export.arxiv.org/api/query?search_query=";
    var first = true;
    
    if (author) {
	if (!first) {
	    baseUrl += '+AND+';
	}
	baseUrl += "au:" + author;
	first = false;
    }
    
    if (title) {
	if (!first) {
	    baseUrl += '+AND+';
	}
	baseUrl += "ti:" + title;
	first = false;
    }
    
    if (abstrct) {
	if (!first) {
	    baseUrl += '+AND+';
	}
	baseUrl += "abs:" + abstrct;
	first = false;
    }
    
    if (all) {
	if (!first) {
	    baseUrl += '+AND+';
	}
	baseUrl += "all:" + all;
    }

    var deferred = $.Deferred();
    $.ajax({
        url: baseUrl,
        type: "get",
        dataType: "xml",
        success: function(xml) {
	    var entry = [];
	    $(xml).find('entry').each(function (index) {
		var id = $(this).find('id').text();
		var pub_date = $(this).find('published').text();
		var title = $(this).find('title').text();
		var summary = $(this).find('summary').text();
		var authors = [];
		$(this).find('author').each(function (index) {
		    authors.push($(this).text());
		});
		
		entry.push({'title': title,
			    'link': id,
			    'summary': summary,
			    'date': pub_date,
			    'authors': authors,
				'publicationName': "Computer Methods in Applied Mechanics and Engineering"
			   });
	    });
	    
	    deferred.resolve(entry);
        },
        error: function(status) {
            console.log("request error " + status + " for url: "+baseUrl);
        }
    });
    return deferred.promise();
}


// 调用 EasyScholar API 获取期刊排名信息的函数
function getPublicationRank(publicationName) {
    var url = 'https://www.easyscholar.cc/open/getPublicationRank?secretKey=f292c0ef1e3d40f2a54a826800f4032e&publicationName=' + encodeURIComponent(publicationName);
    var deferred = $.Deferred();

    $.ajax({
        url: url,
        type: 'get',
        dataType: 'json', // 假设API返回JSON格式数据
        success: function(response) {
            if (response.code === 200 && response.msg === 'SUCCESS') {
				var entry = {
					'data': response.data,
					'code': response.code,
					'msg': response.msg,
			    };
				console.log(response.data);
                deferred.resolve(entry);
            } else {
                deferred.reject('Error: ' + response.msg);
            }
        },
        error: function(xhr, status, error) {
            deferred.reject('AJAX Error: ' + error);
        }
    });
    return deferred.promise();
}

// 假设getStyledSpan函数已经定义，如下：
function getStyledSpan(text, backgroundColor) {
	return `<span style="display:inline-block;padding:2px 5px;background-color:${backgroundColor};color:white;border-radius:3px;">${text}</span>`;
  }