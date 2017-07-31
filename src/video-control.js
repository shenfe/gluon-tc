var document = window.document;

var addCssRule = function (selectorString, styleString) {
    if (window.document.getElementsByTagName('style').length === 0) {
        var tempStyle = window.document.createElement('style');
        tempStyle.setAttribute('type', 'text/css');
        window.document.getElementsByTagName('head')[0].appendChild(tempStyle);
    }

    window.document.getElementsByTagName('style')[0].appendChild(window.document.createTextNode(selectorString + '{' + styleString + '}'));
};

var control = function ($video, option) {
    var $container = option.container || $video.parentElement;

    ensure_styles: {
        
    }

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
