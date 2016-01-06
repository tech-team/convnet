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
    }
  });

  grunt.loadNpmTasks('grunt-bower-requirejs');
  grunt.loadNpmTasks('grunt-contrib-requirejs');
  grunt.loadNpmTasks('grunt-contrib-clean');

  grunt.registerTask('buildjs', ['clean', 'bowerRequirejs', 'requirejs']);
  grunt.registerTask('default', []);
};