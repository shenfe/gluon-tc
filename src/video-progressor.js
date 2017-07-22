var progressor = function () {};

if (typeof exports !== 'undefined') {
    if (typeof module !== 'undefined' && module.exports) {
        exports = module.exports = progressor;
    }
} else {
    window.VideoProgressor = progressor;
}
