function dark_on(){
    $('nav').removeClass("navbar-light");
    $('nav').addClass("navbar-dark");
    $('#get-segmentation-btn').removeClass('btn-success');
    $('#get-segmentation-btn').addClass('btn-dark');
}
function dark_off(){
    $('nav').removeClass("navbar-dark");
    $('nav').addClass("navbar-light");
    $('#get-segmentation-btn').removeClass('btn-dark');
    $('#get-segmentation-btn').addClass('btn-success');
}
(function() {
  var darkSwitch = document.getElementById("darkSwitch");
  if (darkSwitch) {
    initTheme();
    darkSwitch.addEventListener("change", function(event) {
      resetTheme();
    });
    function initTheme() {
      var darkThemeSelected =
        localStorage.getItem("darkSwitch") !== null &&
        localStorage.getItem("darkSwitch") === "dark";
      darkSwitch.checked = darkThemeSelected;
      if(darkThemeSelected){
            document.body.setAttribute("data-theme", "dark")
            dark_on();
        }
        else{
            document.body.removeAttribute("data-theme");
            dark_off();
        }
    }
    function resetTheme() {
      if (darkSwitch.checked) {
        document.body.setAttribute("data-theme", "dark");
        dark_on();
        localStorage.setItem("darkSwitch", "dark");
      } else {
        document.body.removeAttribute("data-theme");
        dark_off();
        localStorage.removeItem("darkSwitch");
      }
    }
  }
})();
