var document = window.document;

var addCssRule = function (selectorString, styleString) {
    if (window.document.getElementsByTagName('style').length === 0) {
        var tempStyle = window.document.createElement('style');
        tempStyle.setAttribute('type', 'text/css');
        window.document.getElementsByTagName('head')[0].appendChild(tempStyle);
    }

    window.document.getElementsByTagName('style')[0].appendChild(window.document.createTextNode(selectorString + '{' + styleString + '}'));
};

var control = function (video, option) {
    var $container = option.container || video.parentElement;

    ensure_styles: {
        if (!window.vctrl_global_style_added) {
            window.vctrl_global_style_added = true;
            addCssRule('.video-control', ``);
        }
    }

    ensure_elements: {
        var $controlBar = document.createElement('div');
        $controlBar.className = 'video-control';
        $controlBar.innerHTML = `
            <button type="button" class="play">play</button>
            <input type="range" class="seek" value="0">
            <button type="button" class="mute">mute</button>
            <input type="range" class="volume" min="0" max="1" step="0.1" value="1">
            <button type="button" class="fullscreen">fullscreen</button>
        `;
        $container.appendChild($controlBar);

        var playButton  = $controlBar.querySelector('.play');
        var muteButton  = $controlBar.querySelector('.mute');
        var fullButton  = $controlBar.querySelector('.fullscreen');
        var seekBar     = $controlBar.querySelector('.seek');
        var volumeBar   = $controlBar.querySelector('.volume');
    }

    var _this = this;
    var hideTimer = null;
    var show = function () {
        hideTimer && window.clearTimeout(hideTimer);
        $controlBar.style.display = 'block';
        hideTimer = window.setTimeout(function () {
            $controlBar.style.display = 'none';
        }, 4000);
    };
    var hide = function () {
        hideTimer && window.clearTimeout(hideTimer);
        hideTimer = window.setTimeout(function () {
            $controlBar.style.display = 'none';
        }, 4000);
    };

    bind_events: {
        playButton.addEventListener('click', function (e) {
            if (video.paused) {
                video.play();
                playButton.innerHTML = 'pause';
            } else {
                video.pause();
                playButton.innerHTML = 'play';
            }
        });

        muteButton.addEventListener('click', function (e) {
            if (!video.muted) {
                video.muted = true;
                muteButton.innerHTML = 'unmute';
            } else {
                video.muted = false;
                muteButton.innerHTML = 'mute';
            }
        });

        fullButton.addEventListener('click', function (e) {
            if (video.requestFullscreen) {
                video.requestFullscreen();
            } else if (video.mozRequestFullScreen) {
                video.mozRequestFullScreen(); // Firefox
            } else if (video.webkitRequestFullscreen) {
                video.webkitRequestFullscreen(); // Chrome and Safari
            }
        });

        seekBar.addEventListener('change', function (e) {
            var time = video.duration * (seekBar.value / 100);
            video.currentTime = time;
        });
        seekBar.addEventListener('mousedown', function (e) {
            video.pause();
        });
        seekBar.addEventListener('mouseup', function (e) {
            video.play();
        });

        volumeBar.addEventListener('change', function (e) {
            video.volume = volumeBar.value;
        });

        video.addEventListener('timeupdate', function (e) {
            var value = (100 / video.duration) * video.currentTime;
            seekBar.value = value;
        });
        video.addEventListener('mouseover', function (e) {
            _this.show();
        }, false);
        video.addEventListener('mouseout', function (e) {
            _this.hide();
        }, false);
        video.addEventListener('click', function (e) {
            _this.show();
        }, false);
    }

    this.show = function () {
        show();
    };
    this.hide = function () {
        hide();
    };
};

if (typeof exports !== 'undefined') {
    if (typeof module !== 'undefined' && module.exports) {
        exports = module.exports = control;
    }
} else {
    window.VideoControl = control;
}
