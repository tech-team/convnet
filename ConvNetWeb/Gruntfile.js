module.exports = function(grunt) {
  grunt.initConfig({
    pkg: grunt.file.readJSON('package.json'),
    clean: {
      build: ['dist/'],
      dev: {
        src: ['dist/**/*']
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
          out: 'dist/js/build.min.js',
          include: ['../components/almond/almond.js']
        }
      }
    },

    copy: {
      bower_fonts: {
        files: [
          {
            expand: true,
            flatten: true,
            src: ['static/components/**/dist/fonts/*'],
            dest: 'dist/fonts/',
            filter: 'isFile'
          }
        ]
      },
      app: {
        files: [
          {
            expand: true,
            flatten: true,
            src: ['static/fonts/**/*'],
            dest: 'dist/fonts/'
          },

          {
            expand: true,
            flatten: true,
            src: ['static/img/**/*'],
            dest: 'dist/img/'
          }
        ]
      }
    },

    bower_concat: {
      libs: {
        cssDest: 'dist/css/libs.min.css',
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
        src: ['dist/css/libs.min.css'],
        dest: 'dist/css/libs.min.css',
        options: {
          sourceMap: false
        }
      },
      css: {
        src: ['static/css/**/*.css'],
        dest: 'dist/css/build.min.css'
      }
    },

    watch: {
      options: {
        livereload: true
      },
      js: {
        files: ['static/js/**/*.js', '!static/js/config.js'],
        tasks: [''],
        options: {
          spawn: false
        }
      },
      css: {
        files: ['static/css/**/*.css'],
        tasks: ['build:css'],
        options: {
          spawn: false
        }
      }
    }

  });

  grunt.loadNpmTasks('grunt-bower-requirejs');
  grunt.loadNpmTasks('grunt-bower-concat');
  grunt.loadNpmTasks('grunt-contrib-requirejs');
  grunt.loadNpmTasks('grunt-contrib-clean');
  grunt.loadNpmTasks('grunt-contrib-concat');
  grunt.loadNpmTasks('grunt-contrib-cssmin');
  grunt.loadNpmTasks('grunt-contrib-copy');
  grunt.loadNpmTasks('grunt-contrib-watch');

  grunt.registerTask('build:css', ['bower_concat', 'cssmin']);
  grunt.registerTask('build:js', ['bowerRequirejs', 'requirejs']);
  grunt.registerTask('build', ['copy', 'build:css', 'build:js']);
  grunt.registerTask('default', ['watch']);
};