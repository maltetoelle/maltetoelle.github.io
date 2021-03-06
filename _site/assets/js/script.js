// import {Cookies} from 'js-cookie'

$("a.bibtexs").click(function(evt) {
 evt.preventDefault()
 var child = $(this).parent().parent().children('pre')
 if (child.is(":visible")) {
  child.hide('slow')//fadeOut()
 } else {
  child.show('slow')//fadeIn()
 }
})

// var cookiesClicked = false
// var cookies = false

function getCookieAgreement() {
  var cookiesClicked = Cookies.get('cookiesClicked')
  if (typeof cookiesClicked == 'undefined') {
    $(".cookie-warning").css("bottom", $(".footer").height() + 30);
    $(".cookie-blocker").css({
     "position": "absolute",
     "top": "0",
     "bottom": "0",
     "z-index": "500",
     "width": "100%",
     "height": $(".wrapper").height(),
     "background-color": "gray",
     "opacity": "0.5",
    });

   } else {
    $(".cookie-blocker").css("display", "none");
    $(".cookie-warning").css("display", "none");
   }
}

$("#cookie-yes").click(() => {
 Cookies.set('cookies', 'true', { expires: 30 })
 Cookies.set('cookiesClicked', 'true', { expires: 30 })
 // cookies = true;
 // cookiesClicked = true;
 getCookieAgreement();
})

$("#cookie-no").click(() => {
 // cookiesClicked = true;
 Cookies.set('cookies', 'false', { expires: 30 })
 Cookies.set('cookiesClicked', 'true', { expires: 30 })
 getCookieAgreement();
})

$(document).ready(function () {

    getCookieAgreement();
    // Read the cookie and if it's defined scroll to id
    // var scroll = $.cookie('scroll');
    var scroll = Cookies.get('scroll')
    console.log(scroll)
    if(scroll){
        scrollToID(scroll, 1000);
        // Cookies.remove('scroll')
    }

    // Handle event onclick, setting the cookie when the href != #
    $('.nav-click').find('a').click(function (e) {
        var href = $(this).attr('href');
        const validHrefs = ['/', '#intro', '#pubs', '#posts']
        if (href == "/contact") {

         return False
        } else {
             e.preventDefault();
             if(validHrefs.indexOf(window.location.pathname) == -1){
                 // $.cookie('scroll', href);
                 if (Cookies.get('cookies') == 'true') {
                  Cookies.set('scoll', href)
                 }
                 // scroll = 'have it'
                 window.location.href = window.location.origin;
                 // scrollToID(href, 500);
             }else{
                 scrollToID(href, 500);
             }
       }
    });

    // scrollToID function
    function scrollToID(id, speed) {
        console.log(id)
        var $href = $(id).attr('href');
        var $anchor = $(id).offset();
        if ($href == "#intro") {
            var offset = 110
        } else {
            var offset = 150
        }
        targetOffset = $anchor.top - offset
        $('html,body').animate({ scrollTop: targetOffset }, speed);
        // Cookies.remove('scroll')
    }

});
