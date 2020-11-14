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

var cookiesClicked = false
var cookies = false

function getCookieAgreement() {
  // Cookies.set('name', 'value')
  if (cookiesClicked == false) {
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
 cookies = true;
 cookiesClicked = true;
 getCookieAgreement();
})

$("#cookie-no").click(() => {
 cookiesClicked = true;
 getCookieAgreement();
})

$(document).ready(function () {

    getCookieAgreement();
    // Read the cookie and if it's defined scroll to id
    var scroll = $.cookie('scroll');
    if(scroll){
        scrollToID(scroll, 1000);
        $.removeCookie('scroll');
    }

    // Handle event onclick, setting the cookie when the href != #
    $('.nav-click').find('a').click(function (e) {
        var href = $(this).attr('href');
        if (href == "/contact") {
         return False
        } else {
             e.preventDefault();
             console.log('here')
             // var id = $(this).attr('id')

             console.log(href)
             if(window.location.pathname == "/contact"){
                 $.cookie('scroll', href);
                 scroll = 'have it'
                 window.location.href = window.location.origin;
             }else{
                 scrollToID(href, 500);
             }
       }
    });

    // scrollToID function
    function scrollToID(id, speed) {
        var $href = $(id).attr('href');
        var $anchor = $(id).offset();
        if ($href == "#intro") {
            var offset = 110
        } else {
            var offset = 150
        }
        targetOffset = $anchor.top - offset
        $('html,body').animate({ scrollTop: targetOffset }, speed);

    }

});
