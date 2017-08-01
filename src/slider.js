var document = window.document;

var addCssRule = function (selectorString, styleString) {
    if (window.document.getElementsByTagName('style').length === 0) {
        var tempStyle = window.document.createElement('style');
        tempStyle.setAttribute('type', 'text/css');
        window.document.getElementsByTagName('head')[0].appendChild(tempStyle);
    }

    window.document.getElementsByTagName('style')[0].appendChild(window.document.createTextNode(selectorString + '{' + styleString + '}'));
};

var Slider = function ($container, option) {
    option = option || {};
    $container.innerHTML = `
        <div class="slider-wrap">
            <div class="slider-bar">
                <div class="slider-dot"></div>
            </div>
        </div>
    `;
    addCssRule('.slider-container', '');
    addCssRule('.slider-container-h', '');
    addCssRule('.slider-container-v', '');
    addCssRule('.slider-container .slider-wrap', 'position:relative;');
    addCssRule('.slider-container-h .slider-wrap', 'width:100%;');
    addCssRule('.slider-container-v .slider-wrap', 'height:100%;');
    addCssRule('.slider-bar', 'position:relative;');
    addCssRule('.slider-dot', 'position:absolute;right:0;display:inline-block;transform:translate3d(0,0,0)');
    addCssRule('.slider-container-h .slider-dot', 'transform:translate3d(50%,50%,0)');
    addCssRule('.slider-container-v .slider-dot', 'transform:translate3d(0,-50%,0)');

    var $wrap = $container.querySelector('.slider-wrap');
    var $bar = $container.querySelector('.slider-bar');
    var $dot = $container.querySelector('.slider-dot');

    var type = (option.dir === 'h') ? 'h' : 'v';
    $container.classList.add('slider-container');
    $container.classList.add('slider-container-' + type);
    var length = {
        total: type === 'h' ? $container.clientWidth : $container.clientHeight,
        progress: option.init || 0
    };
    if (type === 'h') {
        $bar.style.width = '' + length.progress + '%';
    } else {
        $bar.style.height = '' + length.progress + '%';
    }
    var offset = {
        x: 0,
        y: 0
    };

    var onchanging = option.onchanging || function () {};
    var onchanged = option.onchanged || function () {};

    var setOffsetX = function (v) {
        if (v < 0) v = 0;
        if (v > length.total) v = length.total;
        offset.x = v;
        length.progress = offset.x / length.total * 100;
        $bar.style.width = '' + length.progress + '%';
        onchanging(length.progress);
    };
    var setOffsetY = function (v) {
        if (v < 0) v = 0;
        if (v > length.total) v = length.total;
        offset.y = v;
        length.progress = offset.y / length.total * 100;
        $bar.style.height = '' + length.progress + '%';
        onchanging(length.progress);
    };

    var touchstartPos = {
        x: null,
        y: null
    };
    var touchPos = {
        x: null,
        y: null
    };
    $dot.addEventListener('touchstart', function (e) {
        var touchobj = e.changedTouches[0];
        touchstartPos.x = touchPos.x = touchobj.pageX;
        touchstartPos.y = touchPos.y = touchobj.pageY;
    }, false);
    $dot.addEventListener('touchmove', function (e) {
        var touchobj = e.changedTouches[0];
        if (type === 'h') {
            setOffsetX(offset.x - touchPos.x + touchobj.pageX);
        } else {
            setOffsetY(offset.y - touchPos.y + touchobj.pageY);
        }
        touchPos.x = touchobj.pageX;
        touchPos.y = touchobj.pageY;
    }, false);
    $dot.addEventListener('touchend', function (e) {
        touchstartPos.x = null;
        touchstartPos.y = null;
        touchPos.x = null;
        touchPos.y = null;
        onchanged(length.progress);
    }, false);

    Object.defineProperty(this, 'progress', {
        get: function () { return length.progress; },
        set: function (v) {
            if (v < 0) v = 0;
            if (v > 100) v = 100;
            if (type === 'h') {
                setOffsetX(offset.x + length.total * v / 100);
            } else {
                setOffsetY(offset.y + length.total * v / 100);
            }
        }
    });
};

if (typeof exports !== 'undefined') {
    if (typeof module !== 'undefined' && module.exports) {
        exports = module.exports = Slider;
    }
} else {
    window.Slider = Slider;
}
