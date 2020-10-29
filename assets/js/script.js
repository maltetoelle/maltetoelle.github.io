$("a.bibtexs").click(function(evt) {
 evt.preventDefault()
 var child = $(this).parent().parent().children('pre')
 if (child.is(":visible")) {
  child.hide('slow')//fadeOut()
 } else {
  child.show('slow')//fadeIn()
 }
})

// $('.nav-bar').find('a').click(function(){
//     console.log(window.location.pathname)
//     if (window.location.pathname == "/contact") {
//      window.location.href = window.location.origin
//     }
//     console.log('fdsjlk')
//     console.log('fdsjlk')
//     console.log('fdsjlk')
//     var $href = $(this).attr('href');
//     var $anchor = $($href).offset();
//     $anchor.top -= 110
//     console.log($anchor)
//     window.scrollTo($anchor.left,$anchor.top);
//     return false;
// });
//
// $('#intro-click').click(function(){
//     var $href = $(this).attr('href');
//     var $anchor = $($href).offset();
//     $anchor.top -= 150
//     window.scrollTo($anchor.left,$anchor.top);
//     return false;
// });

$(document).ready(function () {
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
