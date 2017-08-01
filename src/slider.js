var document = window.document;

var addCssRule = function (selectorString, styleString) {
    if (window.document.getElementsByTagName('style').length === 0) {
        var tempStyle = window.document.createElement('style');
        tempStyle.setAttribute('type', 'text/css');
        window.document.getElementsByTagName('head')[0].appendChild(tempStyle);
    }

    window.document.getElementsByTagName('style')[0].appendChild(window.document.createTextNode(selectorString + '{' + styleString + '}'));
};

var Slider = function () {

};

if (typeof exports !== 'undefined') {
    if (typeof module !== 'undefined' && module.exports) {
        exports = module.exports = Slider;
    }
} else {
    window.Slider = Slider;
}
