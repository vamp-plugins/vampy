
CXXFLAGS	:= -I../vamp-plugin-sdk -O2 -Wall -I/usr/include/python2.5 
#-fvisibility=hidden

vampy.dylib:	PyPlugin.o PyPlugScanner.o pyvamp-main.o Mutex.o
	g++ -shared $^ -o $@ -L../vamp-plugin-sdk/vamp-sdk -lvamp-sdk -dynamiclib -lpython2.5 -lpthread

# Install plugin
#
LIBRARY_PREFIX		:=/Library
INSTALL_DIR			:=$(LIBRARY_PREFIX)/Audio/Plug-Ins/Vamp
PYEXAMPLE_DIR		:='Example VamPy Plugins'
PLUGIN_NAME			:=vampy
PLUGIN_EXT			:=.dylib
	
install:
	mkdir -p $(INSTALL_DIR)
	rm -f $(INSTALL_DIR)/$(PLUGIN_NAME)$(PLUGIN_EXT)
	cp $(PLUGIN_NAME)$(PLUGIN_EXT) $(INSTALL_DIR)/$(PLUGIN_NAME)$(PLUGIN_EXT)	
	#cp $(PYEXAMPLE_DIR)/*.py $(INSTALL_DIR)
	
installplug : install
cleanplug : clean

clean:	
	rm *.o
	rm *$(PLUGIN_EXT)
	
