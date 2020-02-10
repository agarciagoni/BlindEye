"use strict";
// RPC wrapper

document.getElementById("status").innerHTML = "hello"
console.log("TESTING")


jQuery(document).ready(function($){
    var transitionEnd = 'webkitTransitionEnd otransitionend oTransitionEnd msTransitionEnd transitionend';
    var transitionsSupported = ( $('.csstransitions').length > 0 );
    //if browser does not support transitions - use a different event to trigger them
    if( !transitionsSupported ) transitionEnd = 'noTransition';
    
    //should add a loding while the events are organized 

    function SchedulePlan( element ) {
        this.element = element;
        this.timeline = this.element.find('.timeline');
        this.timelineItems = this.timeline.find('li');
        this.timelineItemsNumber = this.timelineItems.length;
        this.timelineStart = getScheduleTimestamp(this.timelineItems.eq(0).text());
        //need to store delta (in our case half hour) timestamp
        this.timelineUnitDuration = getScheduleTimestamp(this.timelineItems.eq(1).text()) - getScheduleTimestamp(this.timelineItems.eq(0).text());

        this.eventsWrapper = this.element.find('.events');
        this.eventsGroup = this.eventsWrapper.find('.events-group');
        this.singleEvents = this.eventsGroup.find('.single-event');
        this.eventSlotHeight = this.eventsGroup.eq(0).children('.top-info').outerHeight();

        this.modal = this.element.find('.event-modal');
        this.modalHeader = this.modal.find('.header');
        this.modalHeaderBg = this.modal.find('.header-bg');
        this.modalBody = this.modal.find('.body'); 
        this.modalBodyBg = this.modal.find('.body-bg'); 
        this.modalMaxWidth = 800;
        this.modalMaxHeight = 480;

        this.animating = false;

        this.initSchedule();
    }

    SchedulePlan.prototype.initSchedule = function() {
        this.scheduleReset();
        this.initEvents();
    };

    SchedulePlan.prototype.scheduleReset = function() {
        var mq = this.mq();
        if( mq == 'desktop' && !this.element.hasClass('js-full') ) {
            //in this case you are on a desktop version (first load or resize from mobile)
            this.eventSlotHeight = this.eventsGroup.eq(0).children('.top-info').outerHeight();
            this.element.addClass('js-full');
            this.placeEvents();
            this.element.hasClass('modal-is-open') && this.checkEventModal();
        } else if(  mq == 'mobile' && this.element.hasClass('js-full') ) {
            //in this case you are on a mobile version (first load or resize from desktop)
            this.element.removeClass('js-full loading');
            this.eventsGroup.children('ul').add(this.singleEvents).removeAttr('style');
            this.eventsWrapper.children('.grid-line').remove();
            this.element.hasClass('modal-is-open') && this.checkEventModal();
        } else if( mq == 'desktop' && this.element.hasClass('modal-is-open')){
            //on a mobile version with modal open - need to resize/move modal window
            this.checkEventModal('desktop');
            this.element.removeClass('loading');
        } else {
            this.element.removeClass('loading');
        }
    };

    SchedulePlan.prototype.initEvents = function() {
        var self = this;

        this.singleEvents.each(function(){
            //create the .event-date element for each event
            var durationLabel = '<span class="event-date">'+$(this).data('start')+' - '+$(this).data('end')+'</span>';
            $(this).children('a').prepend($(durationLabel));

            //detect click on the event and open the modal
            $(this).on('click', 'a', function(event){
                event.preventDefault();
                if( !self.animating ) self.openModal($(this));
            });
        });

        //close modal window
        this.modal.on('click', '.close', function(event){
            event.preventDefault();
            if( !self.animating ) self.closeModal(self.eventsGroup.find('.selected-event'));
        });
        this.element.on('click', '.cover-layer', function(event){
            if( !self.animating && self.element.hasClass('modal-is-open') ) self.closeModal(self.eventsGroup.find('.selected-event'));
        });
    };

    SchedulePlan.prototype.placeEvents = function() {
        var self = this;
        this.singleEvents.each(function(){
            //place each event in the grid -> need to set top position and height
            var start = getScheduleTimestamp($(this).attr('data-start')),
                duration = getScheduleTimestamp($(this).attr('data-end')) - start;

            var eventTop = self.eventSlotHeight*(start - self.timelineStart)/self.timelineUnitDuration,
                eventHeight = self.eventSlotHeight*duration/self.timelineUnitDuration;
            
            $(this).css({
                top: (eventTop -1) +'px',
                height: (eventHeight+1)+'px'
            });
        });

        this.element.removeClass('loading');
    };

    SchedulePlan.prototype.openModal = function(event) {
        var self = this;
        var mq = self.mq();
        this.animating = true;

        //update event name and time
        this.modalHeader.find('.event-name').text(event.find('.event-name').text());
        this.modalHeader.find('.event-date').text(event.find('.event-date').text());
        this.modal.attr('data-event', event.parent().attr('data-event'));

        //update event content
        this.modalBody.find('.event-info').load(event.parent().attr('data-content')+'.html .event-info > *', function(data){
            //once the event content has been loaded
            self.element.addClass('content-loaded');
        });

        this.element.addClass('modal-is-open');

        setTimeout(function(){
            //fixes a flash when an event is selected - desktop version only
            event.parent('li').addClass('selected-event');
        }, 10);

        if( mq == 'mobile' ) {
            self.modal.one(transitionEnd, function(){
                self.modal.off(transitionEnd);
                self.animating = false;
            });
        } else {
            var eventTop = event.offset().top - $(window).scrollTop(),
                eventLeft = event.offset().left,
                eventHeight = event.innerHeight(),
                eventWidth = event.innerWidth();

            var windowWidth = $(window).width(),
                windowHeight = $(window).height();

            var modalWidth = ( windowWidth*.8 > self.modalMaxWidth ) ? self.modalMaxWidth : windowWidth*.8,
                modalHeight = ( windowHeight*.8 > self.modalMaxHeight ) ? self.modalMaxHeight : windowHeight*.8;

            var modalTranslateX = parseInt((windowWidth - modalWidth)/2 - eventLeft),
                modalTranslateY = parseInt((windowHeight - modalHeight)/2 - eventTop);
            
            var HeaderBgScaleY = modalHeight/eventHeight,
                BodyBgScaleX = (modalWidth - eventWidth);

            //change modal height/width and translate it
            self.modal.css({
                top: eventTop+'px',
                left: eventLeft+'px',
                height: modalHeight+'px',
                width: modalWidth+'px',
            });
            transformElement(self.modal, 'translateY('+modalTranslateY+'px) translateX('+modalTranslateX+'px)');

            //set modalHeader width
            self.modalHeader.css({
                width: eventWidth+'px',
            });
            //set modalBody left margin
            self.modalBody.css({
                marginLeft: eventWidth+'px',
            });

            //change modalBodyBg height/width ans scale it
            self.modalBodyBg.css({
                height: eventHeight+'px',
                width: '1px',
            });
            transformElement(self.modalBodyBg, 'scaleY('+HeaderBgScaleY+') scaleX('+BodyBgScaleX+')');

            //change modal modalHeaderBg height/width and scale it
            self.modalHeaderBg.css({
                height: eventHeight+'px',
                width: eventWidth+'px',
            });
            transformElement(self.modalHeaderBg, 'scaleY('+HeaderBgScaleY+')');
            
            self.modalHeaderBg.one(transitionEnd, function(){
                //wait for the  end of the modalHeaderBg transformation and show the modal content
                self.modalHeaderBg.off(transitionEnd);
                self.animating = false;
                self.element.addClass('animation-completed');
            });
        }

        //if browser do not support transitions -> no need to wait for the end of it
        if( !transitionsSupported ) self.modal.add(self.modalHeaderBg).trigger(transitionEnd);
    };

    SchedulePlan.prototype.closeModal = function(event) {
        var self = this;
        var mq = self.mq();

        this.animating = true;

        if( mq == 'mobile' ) {
            this.element.removeClass('modal-is-open');
            this.modal.one(transitionEnd, function(){
                self.modal.off(transitionEnd);
                self.animating = false;
                self.element.removeClass('content-loaded');
                event.removeClass('selected-event');
            });
        } else {
            var eventTop = event.offset().top - $(window).scrollTop(),
                eventLeft = event.offset().left,
                eventHeight = event.innerHeight(),
                eventWidth = event.innerWidth();

            var modalTop = Number(self.modal.css('top').replace('px', '')),
                modalLeft = Number(self.modal.css('left').replace('px', ''));

            var modalTranslateX = eventLeft - modalLeft,
                modalTranslateY = eventTop - modalTop;

            self.element.removeClass('animation-completed modal-is-open');

            //change modal width/height and translate it
            this.modal.css({
                width: eventWidth+'px',
                height: eventHeight+'px'
            });
            transformElement(self.modal, 'translateX('+modalTranslateX+'px) translateY('+modalTranslateY+'px)');
            
            //scale down modalBodyBg element
            transformElement(self.modalBodyBg, 'scaleX(0) scaleY(1)');
            //scale down modalHeaderBg element
            transformElement(self.modalHeaderBg, 'scaleY(1)');

            this.modalHeaderBg.one(transitionEnd, function(){
                //wait for the  end of the modalHeaderBg transformation and reset modal style
                self.modalHeaderBg.off(transitionEnd);
                self.modal.addClass('no-transition');
                setTimeout(function(){
                    self.modal.add(self.modalHeader).add(self.modalBody).add(self.modalHeaderBg).add(self.modalBodyBg).attr('style', '');
                }, 10);
                setTimeout(function(){
                    self.modal.removeClass('no-transition');
                }, 20);

                self.animating = false;
                self.element.removeClass('content-loaded');
                event.removeClass('selected-event');
            });
        }

        //browser do not support transitions -> no need to wait for the end of it
        if( !transitionsSupported ) self.modal.add(self.modalHeaderBg).trigger(transitionEnd);
    }

    SchedulePlan.prototype.mq = function(){
        //get MQ value ('desktop' or 'mobile') 
        var self = this;
        return window.getComputedStyle(this.element.get(0), '::before').getPropertyValue('content').replace(/["']/g, '');
    };

    SchedulePlan.prototype.checkEventModal = function(device) {
        this.animating = true;
        var self = this;
        var mq = this.mq();

        if( mq == 'mobile' ) {
            //reset modal style on mobile
            self.modal.add(self.modalHeader).add(self.modalHeaderBg).add(self.modalBody).add(self.modalBodyBg).attr('style', '');
            self.modal.removeClass('no-transition');    
            self.animating = false; 
        } else if( mq == 'desktop' && self.element.hasClass('modal-is-open') ) {
            self.modal.addClass('no-transition');
            self.element.addClass('animation-completed');
            var event = self.eventsGroup.find('.selected-event');

            var eventTop = event.offset().top - $(window).scrollTop(),
                eventLeft = event.offset().left,
                eventHeight = event.innerHeight(),
                eventWidth = event.innerWidth();

            var windowWidth = $(window).width(),
                windowHeight = $(window).height();

            var modalWidth = ( windowWidth*.8 > self.modalMaxWidth ) ? self.modalMaxWidth : windowWidth*.8,
                modalHeight = ( windowHeight*.8 > self.modalMaxHeight ) ? self.modalMaxHeight : windowHeight*.8;

            var HeaderBgScaleY = modalHeight/eventHeight,
                BodyBgScaleX = (modalWidth - eventWidth);

            setTimeout(function(){
                self.modal.css({
                    width: modalWidth+'px',
                    height: modalHeight+'px',
                    top: (windowHeight/2 - modalHeight/2)+'px',
                    left: (windowWidth/2 - modalWidth/2)+'px',
                });
                transformElement(self.modal, 'translateY(0) translateX(0)');
                //change modal modalBodyBg height/width
                self.modalBodyBg.css({
                    height: modalHeight+'px',
                    width: '1px',
                });
                transformElement(self.modalBodyBg, 'scaleX('+BodyBgScaleX+')');
                //set modalHeader width
                self.modalHeader.css({
                    width: eventWidth+'px',
                });
                //set modalBody left margin
                self.modalBody.css({
                    marginLeft: eventWidth+'px',
                });
                //change modal modalHeaderBg height/width and scale it
                self.modalHeaderBg.css({
                    height: eventHeight+'px',
                    width: eventWidth+'px',
                });
                transformElement(self.modalHeaderBg, 'scaleY('+HeaderBgScaleY+')');
            }, 10);

            setTimeout(function(){
                self.modal.removeClass('no-transition');
                self.animating = false; 
            }, 20);
        }
    };

    var schedules = $('.cd-schedule');
    var objSchedulesPlan = [],
        windowResize = false;
    
    if( schedules.length > 0 ) {
        schedules.each(function(){
            //create SchedulePlan objects
            objSchedulesPlan.push(new SchedulePlan($(this)));
        });
    }

    $(window).on('resize', function(){
        if( !windowResize ) {
            windowResize = true;
            (!window.requestAnimationFrame) ? setTimeout(checkResize) : window.requestAnimationFrame(checkResize);
        }
    });

    $(window).keyup(function(event) {
        if (event.keyCode == 27) {
            objSchedulesPlan.forEach(function(element){
                element.closeModal(element.eventsGroup.find('.selected-event'));
            });
        }
    });

    function checkResize(){
        objSchedulesPlan.forEach(function(element){
            element.scheduleReset();
        });
        windowResize = false;
    }

    function getScheduleTimestamp(time) {
        //accepts hh:mm format - convert hh:mm to timestamp
        time = time.replace(/ /g,'');
        var timeArray = time.split(':');
        var timeStamp = parseInt(timeArray[0])*60 + parseInt(timeArray[1]);
        return timeStamp;
    }

    function transformElement(element, value) {
        element.css({
            '-moz-transform': value,
            '-webkit-transform': value,
            '-ms-transform': value,
            '-o-transform': value,
            'transform': value
        });
    }
});


function invoke_rpc(method, args, timeout, on_done) {
  hide($("#crash"));
  hide($("#timeout"));
  show($("#rpc_spinner"));
  //send RPC with whatever data is appropriate. Display an error message on crash or timeout
  var xhr = new XMLHttpRequest();
  xhr.open("POST", method, true);
  xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
  xhr.timeout = timeout;
  xhr.send(JSON.stringify(args));
  xhr.ontimeout = function () {
    show($("#timeout"));
    hide($("#rpc_spinner"));
    hide($("#crash"));
  };
  xhr.onloadend = function () {
    if (xhr.status === 200) {
      hide($("#rpc_spinner"));
      var result = JSON.parse(xhr.responseText);
      hide($("#timeout"));
      if (typeof (on_done) != "undefined") {
        on_done(result);
      }
    } else {
      show($("#crash"));
    }
  };
}



// Resource load wrapper
function load_resource(name, on_done) {
  var xhr = new XMLHttpRequest();
  xhr.open("GET", name, true);
  xhr.onloadend = function () {
    if (xhr.status === 200) {
      var result = JSON.parse(xhr.responseText);
      on_done(result);
    }
  };
  xhr.send();
}


function hide($object) {
  $object.css({
    display: 'none'
  });
}

function show($object) {
  $object.css({
    display: 'inline-block'
  });
}

// Code that runs first
$(document).ready(function(){
    invoke_rpc( "/restart", {}, 0, function() { init(); } );
});

function restart(){
  invoke_rpc( "/restart", {} );
}

//  LAB CODE

var step = 50;

var ghosting = false;
var debug = false;
var lines = false;

var busy = false;
var emoji;
var path;
var mouse_action = null;
var svg_width;
var svg_height;
var scale_x;
var scale_y;
var last_data;

var tile_size = 1;
var window_size;

var intervalId = null;

var DIRECTION_BUTTONS = [37,38,39,40];
var BUTTON_ON = "mdl-button--colored button-on";
var PATH_DIM = 30;

print(status)


// Towers are hardcoded here.
var TOWER_INFO = {
    'ThriftyZookeeper': {'price': 7, 'texture': '1f46e'},
    'CheeryZookeeper': {'price': 10, 'texture': '1f477'},
    'SpeedyZookeeper': {'price': 9, 'texture': '1f472'},
    'Demon': {'price': 8, 'texture': '1f479'},
    'CrazyZookeeper': {'price': 11, 'texture': '1f61c'},
    'VHS': {'price': 5, 'texture': '1f4fc'},
    'TraineeZookeeper': {'price': 4, 'texture': '1f476'},
}

var status 
$('#status').text("HELLO");

var new_status = document.getElementById("status");
    new_status.innerHTML = "HELLO";

// convert game data into svg
function display(data) {
    // Initialize the background svg.
    init_svg();
    var states = data[0];
    var formations = data[1];
    var money = data[2];
    path = data[3];
    var targets_remaining = data[4];
    var step_num = data[5] + 1;
    
    // Update balance.
    $('#game-stats').text('Overall Money: $' + money + ' - Remaining Lives: ' + targets_remaining);



    // Create the state and reference state.
    var state = states[0];
    var ref_state = states[1];
    $('#game-state').text(state).removeClass("warning");
    $('#game-frame').text('');
    enable_forward_buttons();

    if (ghosting) {  
      if (ref_state) { 
        $('#game-frame').text('; Ref Frame: '+ step_num);
        if (state != ref_state) {
          $('#game-state').text("should be " + ref_state + ", but is " + state + "!").addClass("warning");
        }
      } else {
        // ghost mode must be over
        disable_forward_buttons();
        if (intervalId) { // if it's running, pause it
          handle_pause_button();
        }
      }
    } 

    var showGhosts = $('#show-ghosts').prop('checked');
    var showMain = $('#show-main').prop('checked');
    formations = formations.filter(function(formation) {
      // if it's a ghost and we're showing ghosts or
      // if it's not a ghost and we're showing normals
      return (!!formation.ghost && showGhosts) || (!formation.ghost && showMain);
    });

    // build list of svg for emoji
    var flist = [];
    formations.forEach(function(formation) {
        // formation attributes: texture, rect
        var svg = emoji[formation.texture];
        if (svg === undefined) {
            console.log('no emoji for '+JSON.stringify(formation));
        } else {
            var x = (formation.rect[0]-(formation.rect[2])/2).toString();
            var y = (formation.rect[1]-(formation.rect[3])/2).toString();
            var g = '<g transform="translate('+x+' '+y+')';
            
            var scalex = scale_x * ((formation.rect[2]) / 60);
            var scaley = scale_y * ((formation.rect[3]) / 60);
            
            if (scalex != 1 || scaley != 1) {
                g += 'scale('+scalex.toString()+' '+scaley.toString()+')';
            }
            g += '"';
            if (formation.ghost) {
                g += ' opacity="0.5"';
            }
            g += '>';
            g += svg;
            if (debug) {
                g += '<rect x="0" y="0" width="64" height="64" stroke="red" stroke-width="2" fill="none"/>';
                g += '<text x="-5" y="-5" text-anchor="end" stroke="red" fill="red">'+x.toString()+","+y.toString()+'</text>';
            }
            g += '</g>';
            flist.push(g);

            // Line of sight
            if (lines && formation.aim_dir !== undefined && formation.aim_dir !== null) {
                var center_x = (formation.rect[0]).toString();
                var center_y = (formation.rect[1]).toString();
                var end_x = formation.aim_dir[0] * 3000 // long enough to cover whole board
                var end_y = formation.aim_dir[1] * 3000 
                var g = '<g transform="translate('+center_x+' '+center_y+')';
                if (scalex != 1 || scaley != 1) {
                    g += 'scale('+scalex.toString()+' '+scaley.toString()+')';
                }
                g += '" opacity="0.4">';
                g += '<line x1="0" y1="0" x2="'+end_x+'" y2="'+end_y+'" stroke="red" stroke-width="4" fill="none"/>';
                g += '</g>';
                flist.push(g);
            }
        }
    });

    // Update the SVG.
    var bigger_svg = document.getElementById("game-grid");
    print(status)
    bigger_svg.innerHTML = "HELLO";
}

function init_svg() {
    var w = $('#wrapper');
    svg_width = w.width();
    svg_height = 3*svg_width/4;   // 4:3 aspect ratio
    
    // Initialize the global scale.
    scale_x = svg_width / window_size[0];
    scale_y = svg_height / window_size[1];
}

function create_path(path) {
    var plist = [];
    for (var i = 0; i < path.length-1; i++) {
        // Extract meaningful information from the points.
        var start_x = path[i][0], start_y = path[i][1];
        var end_x = path[i+1][0], end_y = path[i+1][1];
        
        // Depending on what direction is being traversed, construct the rectangle.
        var x, y, width, height;
        if (start_x == end_x) {
            width = PATH_DIM;
            height = Math.abs(end_y-start_y)+PATH_DIM;
            
            // Going down.
            if (end_y > start_y) {
                x = (start_x-PATH_DIM/2).toString();
                y = (start_y-PATH_DIM/2).toString();
            } else {
                x = (end_x-PATH_DIM/2).toString();
                y = (end_y-PATH_DIM/2).toString();
            }
        } else {
            width = Math.abs(end_x-start_x)+PATH_DIM;
            height = PATH_DIM;
            
            // Going right.
            if (start_x < end_x) {
                x = (start_x-PATH_DIM/2).toString();
                y = (start_y-PATH_DIM/2).toString();
            } else {
                x = (end_x-PATH_DIM/2).toString();
                y = (end_y-PATH_DIM/2).toString();
            }
        }
        
        // Construct the rectangle.
        var g = "hello"
        //var g = '<rect x="' + x + '" y="' + y + '" width="' + width + '" height="' + height + '" stroke="#e4e4a1" stroke-width="1" fill="#e4e4a1"/>';
        plist.push(g);
    }
    //return plist.join('');
    return status
}

function debug_render() {
  if (last_data) {
    display(last_data);
  }
}

function disable_ghost_button(){
  $("#ghost").prop('disabled', true).css('visibility', 'hidden');
}

function enable_ghost_button(){
  $("#ghost").prop('disabled', false).css('visibility', 'visible');
}

function disable_forward_buttons(){
  $("#step_simulation").prop('disabled', true);
  $("#run_simulation").prop('disabled', true);
}

function enable_forward_buttons(){
  $("#step_simulation").prop('disabled', false);
  $("#run_simulation").prop('disabled', false);
}

function hide_all_simulate_buttons() {
  $("#step_simulation").css('display','none');
  $("#run_simulation").css('display','none');
  $("#pause_simulation").css('display','none');
}

function show_forward_buttons() {
  $("#pause_simulation").css('display','none');
  $("#run_simulation").css('display','inline-block');
  if (ghosting) {
    $("#step_simulation").css('display','inline-block');
  }
}

function show_pause_button() {
  $("#pause_simulation").css('display','inline-block');
  $("#run_simulation").css('display','none');
  $("#step_simulation").css('display','none');
}

function timestep(actions) {
    busy = true;

    init_svg();

    var send_action = mouse_action;
    mouse_action = null;
    invoke_rpc('/timestep', [send_action, ghosting], 1000, function (data) {
        last_data = data;
        if (emoji) display(data);
        busy = false;
    });
}

// like timestep, but don't advance game state
function render() {
    busy = true;
    init_svg();

    invoke_rpc('/render', [ghosting], 1000, function (data) {
        last_data = data;
        if (emoji) display(data);
        busy = false;
    });
}

function init_gui() {
    // add mouse listener to game board
    mouse_action = null;
    
    $("#wrapper").click(function(event) {
        var posX = $(this).offset().left, posY = $(this).offset().top;
        mouse_action = [(event.pageX - posX), (event.pageY - posY)];
    });
    
    $('#show-main').on('change', debug_render);
    $('#show-ghosts').on('change', debug_render);
    
    // load SVG for all the emoji
    load_resource('/resources/emoji.json',function (data) {
        emoji = {};
        var re = new RegExp('\<svg.*?\>(.*)\</svg\>');
        $.each(data,function(codepoint,svg) {
            svg = svg.replace(re,'$1');
            emoji[codepoint] = svg;
        });
        $(document).ready(function(){
            init_tower_carousel();
            $(".owl-carousel").owlCarousel({
                loop: false,
                margin: 10,
                nav: true,
                navText: ["<img src='left-arrow.png' class='nav-arrow'>","<img src='right-arrow.png' class='nav-arrow'>"],
                mouseDrag: false
            });
        });  
        if (last_data) display(last_data);
    });

    // hide controls until we have a map
    hide_all_simulate_buttons();
    
    // getting around a material bug where the menu doesn't close on click?
    $('#map_list').click(function () {
      $('.is-visible').removeClass('is-visible');
    });

    // set up map selection for maps
    invoke_rpc("/ls", {"path":"resources/maps/"}, 0, function(loaded) {
        loaded.sort();
        for (var i in loaded) {
            if (loaded[i] != "zoo1-tiny.json") { // do not display zoo1-tiny on UI
                $("#map_list").append(
                    "<li class=\"mdl-menu__item\" onclick=\"handle_map_select('" +
                        loaded[i] +
                        "')\">" +
                        loaded[i] +
                        "</li>");
            }
        }
    });

    // set up map selection for test cases
    invoke_rpc("/ls", {"path":"cases/"}, 0, function(loaded) {
        loaded.sort();
        let firstMap = null;
        let badMaps = new Set(['zoo1-tiny.json'])
        for (var i in loaded) {
            let testClass = parseInt(loaded[i].split("-")[0]);
            //Booleans:
            let showTestClass = (testClass !== undefined);
            let isGoodMap = !badMaps.has(loaded[i]);
            let isInFile = loaded[i].endsWith(".in");

            if (showTestClass && isGoodMap && isInFile) {
                if (firstMap === null) firstMap = loaded[i];
                $("#map_list").append(
                    "<li class=\"mdl-menu__item\" onclick=\"handle_map_select('" +
                        loaded[i] +
                        "')\">" +
                        loaded[i] +
                        "</li>");
            }
        }
        
        // If we wanted to select a map:
        // start by selecting a map
        // if a valid one is stored, us it
        // var map = sessionStorage.getItem('map');
        // if (!map || loaded.indexOf(map)<0) {
        //   map = firstMap;
        // }
        // handle_map_select(map);
    });
}

function init_tower_carousel() {
    // Fetch the carousel.
    var tower_carousel = document.getElementById("tower-carousel");
    
    // Create divs for each item.
    for (let key in TOWER_INFO) {
        // Create the containing div and store attributes.
        let div = document.createElement('div');
        div.setAttribute("id", key);
        div.classList.add("tower");
        div.classList.add("item");
        let new_emoji = emoji[TOWER_INFO[key]['texture']];
        if (new_emoji === undefined) {
            console.log('no emoji for '+JSON.stringify(key));
        }
        let tower_cost = TOWER_INFO[key]['price'];
        
        // Create the SVG (icon) of the tower.
        let body = '<svg width="' + 
            "60" +
            '" height="' +
            "60"+
            '" viewbox="0 0 ' +
            "60"+
            ' ' +
            "60"+
            '">';
        body += '<g>' + new_emoji + '</g>'
        body += "</svg>";
        
        // Add text.
        body += '<p>' + key.toString() + '</p>'
        body += '<p>$' + tower_cost.toString() + '</p>'
        
        // Add all information to the tower. the tower.
        div.innerHTML = status;
        
        // Add listeners.
        div.addEventListener('click', function(event) {
            mouse_action = key;
            /*
            $('.item').removeClass('selected');
            $('#'+key).addClass('selected').focus();
            */
        });
        
        $(div).click(function(event) {
            mouse_action = key;
        });


        tower_carousel.appendChild(div);
    }
}

function handle_map_select(value){
    pause();
    ghosting = false;
    update_ghost_button_display();

    hide_all_simulate_buttons();

    invoke_rpc('/init_game',value,1000,function (args) {
        $('#current_map').text(value);
        show_forward_buttons();
        sessionStorage.setItem('map', value);
        if (args[0]) {
          enable_ghost_button();
        } else {
          disable_ghost_button();
        }
        window_size = args[1];
        
        render();
    });
}

function handle_reset_button() {
    pause();
    handle_map_select($('#current_map').text());
}

function handle_simulate_button(){
  // start simulation
  if(!intervalId){
    // show / hide GUI elements
    show_pause_button();
    mouse_action = null;
    start();
  }
}

function handle_step_button(){
  timestep();
}

function handle_ghost_button(){
    ghosting = !ghosting;
    update_ghost_button_display();
    render();
}

function update_ghost_button_display() {
    var button = $('#ghost');
    var toggles = $('.view-toggle');
    var step_button = $('#step_simulation');
    if (ghosting) {
        button.addClass(BUTTON_ON);
        toggles.css('visibility', 'visible');
        step_button.css('display','inline-block');
    } else {
        button.removeClass(BUTTON_ON);
        toggles.css('visibility', 'hidden');
        step_button.css('display','none');
    }
  
}

function handle_pause_button(){
  if(intervalId){
    // show / hide GUI elements
    show_forward_buttons();
    pause();
  }
}

function handle_debug_button() {
    debug = !debug;
    var button = $('#debug');
    if (debug) {
        button.addClass(BUTTON_ON);
    } else {
        button.removeClass(BUTTON_ON)
    }

    if (last_data) display(last_data);
}

function handle_line_button() {
    lines = !lines;
    var button = $('#debug2');
    if (lines) {
        button.addClass(BUTTON_ON);
    } else {
        button.removeClass(BUTTON_ON)
    }

    if (last_data) display(last_data);
}

function start() {
    timestep();
    intervalId = setInterval(function() {
      if (!busy) timestep()
    }, step);
}

function pause() {
    clearInterval(intervalId);
    intervalId = null;
}

function init(){
    init_gui();
}


