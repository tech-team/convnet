module.exports = function(grunt) {
  grunt.initConfig({
    pkg: grunt.file.readJSON('package.json'),
    clean: {
      build: ['static/build/'],
      dev: {
        src: ['static/build/**/*']
      }
    },

    bowerRequirejs: {
      target: {
        rjsConfig: 'static/js/config.js',
        options: {
          exclude: ['requirejs', 'almond']
        }
      }
    },

    requirejs: {
      compile: {
        options: {
          almond: true,
          baseUrl: 'static/js',
          mainConfigFile: 'static/js/config.js',
          name: 'config',
          optimize: 'uglify2',
          mangle: false,
          generateSourceMaps: true,
          preserveLicenseComments: false,
          out: 'static/build/build.min.js',
          include: ['../../bower_components/almond/almond.js']
        }
      }
    },

    bower_concat: {
      libs: {
        cssDest: 'static/build/libs.min.css',
        exclude: [
          'almond',
          'requirejs',
          'jquery',
          'lodash'
        ],
        mainFiles: {
          bootstrap: ['dist/css/bootstrap.css']
        },
        dependencies: {
          //'backbone': 'underscore',
          //'jquery-mousewheel': 'jquery'
        },
        bowerOptions: {
          //relative: false
        }
      }
    },

    cssmin: {
      options: {
        shorthandCompacting: false,
        roundingPrecision: -1,
        sourceMap: true
      },
      libs: {
        src: ['static/build/libs.min.css'],
        dest: 'static/build/libs.min.css',
        options: {
          sourceMap: false
        }
      },
      css: {
        src: ['static/css/**/*.css'],
        dest: 'static/build/build.min.css'
      }
    }

  });

  grunt.loadNpmTasks('grunt-bower-requirejs');
  grunt.loadNpmTasks('grunt-bower-concat');
  grunt.loadNpmTasks('grunt-contrib-requirejs');
  grunt.loadNpmTasks('grunt-contrib-clean');
  grunt.loadNpmTasks('grunt-contrib-concat');
  grunt.loadNpmTasks('grunt-contrib-cssmin');

  grunt.registerTask('build:css', ['bower_concat', 'cssmin']);
  grunt.registerTask('build:js', ['bowerRequirejs', 'requirejs']);
  grunt.registerTask('build', ['build:css', 'build:js']);
  grunt.registerTask('default', []);
};