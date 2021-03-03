/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var t=function(b){var d=0;return function(){return d<b.length?{done:!1,value:b[d++]}:{done:!0}}},u=function(b){var d="undefined"!=typeof Symbol&&Symbol.iterator&&b[Symbol.iterator];return d?d.call(b):{next:t(b)}},v=function(b){b=["object"==typeof globalThis&&globalThis,b,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var d=0;d<b.length;++d){var h=b[d];if(h&&h.Math==Math)return h}throw Error("Cannot find global object");},w=v(this),x="function"==typeof Object.defineProperties?
Object.defineProperty:function(b,d,h){if(b==Array.prototype||b==Object.prototype)return b;b[d]=h.value;return b},z=function(b,d){if(d)a:{var h=w;b=b.split(".");for(var k=0;k<b.length-1;k++){var e=b[k];if(!(e in h))break a;h=h[e]}b=b[b.length-1];k=h[b];d=d(k);d!=k&&null!=d&&x(h,b,{configurable:!0,writable:!0,value:d})}};
z("Promise",function(b){function d(){this.a=null}function h(a){return a instanceof e?a:new e(function(c){c(a)})}if(b)return b;d.prototype.b=function(a){if(null==this.a){this.a=[];var c=this;this.c(function(){c.g()})}this.a.push(a)};var k=w.setTimeout;d.prototype.c=function(a){k(a,0)};d.prototype.g=function(){for(;this.a&&this.a.length;){var a=this.a;this.a=[];for(var c=0;c<a.length;++c){var f=a[c];a[c]=null;try{f()}catch(g){this.f(g)}}}this.a=null};d.prototype.f=function(a){this.c(function(){throw a;
})};var e=function(a){this.b=0;this.g=void 0;this.a=[];var c=this.c();try{a(c.resolve,c.reject)}catch(f){c.reject(f)}};e.prototype.c=function(){function a(g){return function(l){f||(f=!0,g.call(c,l))}}var c=this,f=!1;return{resolve:a(this.o),reject:a(this.f)}};e.prototype.o=function(a){if(a===this)this.f(new TypeError("A Promise cannot resolve to itself"));else if(a instanceof e)this.s(a);else{a:switch(typeof a){case "object":var c=null!=a;break a;case "function":c=!0;break a;default:c=!1}c?this.m(a):
this.i(a)}};e.prototype.m=function(a){var c=void 0;try{c=a.then}catch(f){this.f(f);return}"function"==typeof c?this.u(c,a):this.i(a)};e.prototype.f=function(a){this.j(2,a)};e.prototype.i=function(a){this.j(1,a)};e.prototype.j=function(a,c){if(0!=this.b)throw Error("Cannot settle("+a+", "+c+"): Promise already settled in state"+this.b);this.b=a;this.g=c;this.l()};e.prototype.l=function(){if(null!=this.a){for(var a=0;a<this.a.length;++a)y.b(this.a[a]);this.a=null}};var y=new d;e.prototype.s=function(a){var c=
this.c();a.h(c.resolve,c.reject)};e.prototype.u=function(a,c){var f=this.c();try{a.call(c,f.resolve,f.reject)}catch(g){f.reject(g)}};e.prototype.then=function(a,c){function f(m,n){return"function"==typeof m?function(p){try{g(m(p))}catch(q){l(q)}}:n}var g,l,r=new e(function(m,n){g=m;l=n});this.h(f(a,g),f(c,l));return r};e.prototype.catch=function(a){return this.then(void 0,a)};e.prototype.h=function(a,c){function f(){switch(g.b){case 1:a(g.g);break;case 2:c(g.g);break;default:throw Error("Unexpected state: "+
g.b);}}var g=this;null==this.a?y.b(f):this.a.push(f)};e.resolve=h;e.reject=function(a){return new e(function(c,f){f(a)})};e.race=function(a){return new e(function(c,f){for(var g=u(a),l=g.next();!l.done;l=g.next())h(l.value).h(c,f)})};e.all=function(a){var c=u(a),f=c.next();return f.done?h([]):new e(function(g,l){function r(p){return function(q){m[p]=q;n--;0==n&&g(m)}}var m=[],n=0;do m.push(void 0),n++,h(f.value).h(r(m.length-1),l),f=c.next();while(!f.done)})};return e});
var A=this||self,B=/^[\w+/_-]+[=]{0,2}$/,C=null,D=function(){var b=A.document;return(b=b.querySelector&&b.querySelector("script[nonce]"))&&(b=b.nonce||b.getAttribute("nonce"))&&B.test(b)?b:""};function E(){window.performance.mark("gapi_load_end");window.performance.measure("gapi_load","gapi_load_start","gapi_load_end");F()}var F,G=null;
function H(){if(G)return G;window.performance.mark("gapi_load_start");var b=window.colabExperiments&&window.colabExperiments.first_party_auth;return G=new Promise(function(d,h){F=d;window.gapi_onload=function(){b?F():gapi.load("auth:client",E)};var k=document.createElement("script");k.src="https://apis.google.com/js/client.js";k.async=!0;null===C&&(C=D());(d=C)&&k.setAttribute("nonce",d);k.onerror=function(){h(Error("Error loading "+k.src))};d=document.getElementsByTagName("script")[0];d.parentNode.insertBefore(k,
d)})}window.colab_gapi_loader={};window.colab_gapi_loader.load=H;H().then(function(){},function(){});