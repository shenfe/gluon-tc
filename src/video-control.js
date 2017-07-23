var document = window.document;

var control = function ($video, option) {
    var $container = option.container || $video.parentElement;

    ensure_elements: {
        var $progressBar = document.createElement('div');
        var $progressDot = document.createElement('div');
        $progressBar.appendChild($progressDot);
        $container.appendChild($progressBar);
    }

    bind_events: {
        $progressDot.addEventListener('touchstart', function (e) {}, false);
        $progressDot.addEventListener('touchmove', function (e) {}, false);
        $progressDot.addEventListener('touchend', function (e) {}, false);
        $container.addEventListener('click', function (e) {}, false);
    }
};

if (typeof exports !== 'undefined') {
    if (typeof module !== 'undefined' && module.exports) {
        exports = module.exports = control;
    }
} else {
    window.VideoControl = control;
}
