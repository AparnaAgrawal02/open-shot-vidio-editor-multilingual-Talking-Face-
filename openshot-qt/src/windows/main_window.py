"""
 @file
 @brief This file loads the main window (i.e. the primary user-interface)
 @author Noah Figg <eggmunkee@hotmail.com>
 @author Jonathan Thomas <jonathan@openshot.org>
 @author Olivier Girard <olivier@openshot.org>

 @section LICENSE

 Copyright (c) 2008-2018 OpenShot Studios, LLC
 (http://www.openshotstudios.com). This file is part of
 OpenShot Video Editor (http://www.openshot.org), an open-source project
 dedicated to delivering high quality video editing and animation solutions
 to the world.

 OpenShot Video Editor is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 OpenShot Video Editor is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with OpenShot Library.  If not, see <http://www.gnu.org/licenses/>.
 """

import os
import shutil
import webbrowser
from copy import deepcopy
from time import sleep
from uuid import uuid4
import sys
import time

# my imports

import random, string

##############

import openshot  # Python module for libopenshot (required video editing module installed separately)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QCoreApplication, PYQT_VERSION_STR,
    QTimer, QDateTime, QFileInfo, QUrl, QThread, QObject
    )
from PyQt5.QtGui import QIcon, QCursor, QKeySequence, QTextCursor, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QDockWidget,
    QMessageBox, QDialog, QFileDialog, QInputDialog,
    QAction, QActionGroup, QSizePolicy,
    QStatusBar, QToolBar, QToolButton,
    QLineEdit, QComboBox, QTextEdit, QMenu, QPushButton, QLabel
)

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from classes import exceptions, info, qt_types, ui_util, updates
from classes.app import get_app
from classes.exporters.edl import export_edl
from classes.exporters.final_cut_pro import export_xml
from classes.importers.edl import import_edl
from classes.importers.final_cut_pro import import_xml
from classes.logger import log
from classes.metrics import track_metric_session, track_metric_screen
from classes.query import Clip, Transition, Marker, Track
from classes.thumbnail import httpThumbnailServerThread
from classes.time_parts import secondsToTimecode
from classes.timeline import TimelineSync
from classes.version import get_current_Version
from windows.models.effects_model import EffectsModel
from windows.models.emoji_model import EmojisModel
from windows.models.files_model import FilesModel
from windows.models.transition_model import TransitionsModel
from windows.preview_thread import PreviewParent
from windows.video_widget import VideoWidget
from windows.views.effects_listview import EffectsListView
from windows.views.effects_treeview import EffectsTreeView
from windows.views.emojis_listview import EmojisListView
from windows.views.files_listview import FilesListView
from windows.views.files_treeview import FilesTreeView
from windows.views.properties_tableview import PropertiesTableView, SelectionLabel
from windows.views.webview import TimelineWebView
from windows.views.transitions_listview import TransitionsListView
from windows.views.transitions_treeview import TransitionsTreeView
from windows.views.tutorial import TutorialManager

# print(sys.path)
# from TTS import inference
# from Wav2Lip import inference_class
# from syncnet_python import syncnet_infer
# import inference
import subprocess
# from slides_ocr import slide_class
# import platform
import cv2
import numpy as np

ref_point = []

# class SlideTranslate2(QRunnable):
#     def __init__(self, progress, ocr, text):
#         QRunnable.__init__(self)
#         self.w = progress
#         self.files_model = FilesModel()
#         self.ocr = ocr
#         self.text = text

#     def run(self):

#         log.info("Worker actionSlideTranslate2_trigger")
#         self.text = self.text.split("\n")
#         self.ocr.textUpdate(self.text)
#         qfilename = QUrl.fromLocalFile(self.ocr.outfile)
#         self.files_model.process_urls([qfilename])

#         QMetaObject.invokeMethod(self.w, "setValue",
#             Qt.QueuedConnection, Q_ARG(int, 1))

class FOMM(QRunnable):
    def __init__(self, pos, url_fomm, progress, pth):
        QRunnable.__init__(self)
        self.pos = pos
        self.url_fomm = url_fomm
        self.w = progress
        self.files_model = FilesModel()
        self.pth = pth

    def run(self):

        log.info("Worker actionFOMM_trigger")

        name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
        ls_pos = sorted(self.pos['ls'])
        
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])

        ls_pos = [int(fps_float * x + 1) for x in ls_pos[1:-1]]
        # print(ls_pos)
        from windows.export import Export
        win = Export()
        print(ls_pos, name)
        print(ls_pos)
        fn, epn = win.accept(ls_pos[0], ls_pos[1], True, name)
        print(ls_pos, name, fn, epn)

        outdir = os.path.join(sys.path[0], 'ls_out', 'to_send')
        pth1 = os.path.join(outdir, name + '.mp4')
        video_cmd = 'ffmpeg -i {} -c copy -an {}'.format(epn, pth1)

        subprocess.call(video_cmd, shell=True)

        outfile = os.path.join(sys.path[0], 'ls_out', 'res', name + '.mp4')

        cmd = 'curl -X POST -F file1=@{} -F file2=@{} {} > {}'.format(self.pth, pth1, self.url_fomm, outfile)
        
        subprocess.call(cmd, shell=True)
        qfilename = QUrl.fromLocalFile(outfile)
        self.files_model.process_urls([qfilename])

        QMetaObject.invokeMethod(self.w, "setValue",
            Qt.QueuedConnection, Q_ARG(int, 1))

class MakeItTalk(QRunnable):
    def __init__(self, pos, url_mit, progress, pth):
        QRunnable.__init__(self)
        self.pos = pos
        self.url_mit = url_mit
        self.w = progress
        self.files_model = FilesModel()
        self.pth = pth

    def run(self):

        log.info("Worker actionMakeItTalk_trigger")

        name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
        ls_pos = sorted(self.pos['ls'])
        
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])

        ls_pos = [int(fps_float * x + 1) for x in ls_pos[1:-1]]
        # print(ls_pos)
        from windows.export import Export
        win = Export()
        #print(ls_pos, name)
        print(ls_pos)
        fn, epn = win.accept(ls_pos[0], ls_pos[1], True, name)
        #print(ls_pos, name, fn, epn,"letssgggoooo")

        outdir = os.path.join(sys.path[0], 'ls_out', 'to_send')
        pth2 = os.path.join(outdir, name + '.wav')
        audio_cmd = 'ffmpeg -i {} {}'.format(epn, pth2)
        print(audio_cmd )
        subprocess.call(audio_cmd, shell=True)

        outfile = os.path.join(sys.path[0], 'ls_out', 'res', name + '.mp4')

        cmd = 'curl -X POST -F file1=@{} -F file2=@{} {} > {}'.format(self.pth, pth2, self.url_mit, outfile)
        print(cmd)
        subprocess.call(cmd, shell=True)
        qfilename = QUrl.fromLocalFile(outfile)
        self.files_model.process_urls([qfilename])

        QMetaObject.invokeMethod(self.w, "setValue",
            Qt.QueuedConnection, Q_ARG(int, 1))
        print("OKAY")

class MainWindow(updates.UpdateWatcher, QMainWindow):
    """ This class contains the logic for the main window widget """

    # Path to ui file
    ui_path = os.path.join(info.PATH, 'windows', 'ui', 'main-window.ui')

    previewFrameSignal = pyqtSignal(int)
    refreshFrameSignal = pyqtSignal()
    refreshFilesSignal = pyqtSignal()
    refreshTransitionsSignal = pyqtSignal()
    LoadFileSignal = pyqtSignal(str)
    PlaySignal = pyqtSignal(int)
    PauseSignal = pyqtSignal()
    StopSignal = pyqtSignal()
    SeekSignal = pyqtSignal(int)
    SpeedSignal = pyqtSignal(float)
    RecoverBackup = pyqtSignal()
    FoundVersionSignal = pyqtSignal(str)
    WaveformReady = pyqtSignal(str, list)
    TransformSignal = pyqtSignal(str)
    SelectRegionSignal = pyqtSignal(str)
    MaxSizeChanged = pyqtSignal(object)
    InsertKeyframe = pyqtSignal(object)
    OpenProjectSignal = pyqtSignal(str)
    ThumbnailUpdated = pyqtSignal(str)
    FileUpdated = pyqtSignal(str)
    CaptionTextUpdated = pyqtSignal(str, object)
    CaptionTextLoaded = pyqtSignal(str, object)
    TimelineZoom = pyqtSignal(float)     # Signal to zoom into timeline from zoom slider
    TimelineScrolled = pyqtSignal(list)  # Scrollbar changed signal from timeline
    TimelineScroll = pyqtSignal(float)   # Signal to force scroll timeline to specific point
    TimelineCenter = pyqtSignal()        # Signal to force center scroll on playhead
    SelectionAdded = pyqtSignal(str, str, bool)  # Signal to add a selection
    SelectionRemoved = pyqtSignal(str, str)      # Signal to remove a selection
    SelectionChanged = pyqtSignal()      # Signal after selections have been changed (added/removed)

    # Docks are closable, movable and floatable
    docks_frozen = False

    # Save window settings on close
    def closeEvent(self, event):

        app = get_app()
        # Some window managers handels dragging of the modal messages incorrectly if other windows are open
        # Hide tutorial window first
        self.tutorial_manager.hide_dialog()

        # Prompt user to save (if needed)
        if app.project.needs_save() and self.mode != "unittest":
            log.info('Prompt user to save project')
            # Translate object
            _ = app._tr

            # Handle exception
            ret = QMessageBox.question(
                self,
                _("Unsaved Changes"),
                _("Save changes to project before closing?"),
                QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                # Save project
                self.actionSave_trigger()
                event.accept()
            elif ret == QMessageBox.Cancel:
                # Show tutorial again, if any
                self.tutorial_manager.re_show_dialog()
                # User canceled prompt - don't quit
                event.ignore()
                return

        # Log the exit routine
        log.info('---------------- Shutting down -----------------')
        cv2.destroyAllWindows()
        # Close any tutorial dialogs
        self.tutorial_manager.exit_manager()

        # Save settings
        self.save_settings()

        # Track end of session
        track_metric_session(False)

        # Stop threads
        self.StopSignal.emit()

        # Process any queued events
        QCoreApplication.processEvents()

        # Stop preview thread (and wait for it to end)
        self.preview_thread.player.CloseAudioDevice()
        self.preview_thread.kill()
        self.preview_parent.background.exit()
        self.preview_parent.background.wait(5000)

        # Close Timeline
        self.timeline_sync.timeline.Close()
        self.timeline_sync.timeline = None

        # Destroy lock file
        self.destroy_lock_file()

    def recover_backup(self):
        """Recover the backup file (if any)"""
        log.info("recover_backup")

        # Check for backup.osp file
        if os.path.exists(info.BACKUP_FILE):
            # Load recovery project
            log.info("Recovering backup file: %s" % info.BACKUP_FILE)
            self.open_project(info.BACKUP_FILE, clear_thumbnails=False)

            # Clear the file_path (which is set by saving the project)
            project = get_app().project
            project.current_filepath = None
            project.has_unsaved_changes = True

            # Set Window title
            self.SetWindowTitle()

            # Show message to user
            msg = QMessageBox()
            _ = get_app()._tr
            msg.setWindowTitle(_("Backup Recovered"))
            msg.setText(_("Your most recent unsaved project has been recovered."))
            msg.exec_()

        else:
            # No backup project found
            # Load a blank project (to propagate the default settings)
            get_app().project.load("")
            self.actionUndo.setEnabled(False)
            self.actionRedo.setEnabled(False)
            self.SetWindowTitle()

    def create_lock_file(self):
        """Create a lock file"""
        lock_path = os.path.join(info.USER_PATH, ".lock")
        # Check if it already exists
        if os.path.exists(lock_path):
            exceptions.libopenshot_crash_recovery()
            log.error("Unhandled crash detected. Preserving cache.")
            self.destroy_lock_file()
        else:
            # Normal startup, clear thumbnails
            self.clear_all_thumbnails()

        # Write lock file (try a few times if failure)
        lock_value = str(uuid4())
        for attempt in range(5):
            try:
                # Create lock file
                with open(lock_path, 'w') as f:
                    f.write(lock_value)
                log.debug("Wrote value %s to lock file %s", lock_value, lock_path)
                break
            except OSError:
                log.debug("Failed to write lock file (attempt: %d)", attempt, exc_info=1)
                sleep(0.25)

    def destroy_lock_file(self):
        """Destroy the lock file"""
        lock_path = os.path.join(info.USER_PATH, ".lock")

        # Remove file (try a few times if failure)
        for attempt in range(5):
            try:
                os.remove(lock_path)
                log.debug("Removed lock file {}".format(lock_path))
                break
            except FileNotFoundError:
                break
            except OSError:
                log.debug('Failed to destroy lock file (attempt: %s)' % attempt, exc_info=1)
                sleep(0.25)

    def actionNew_trigger(self):

        app = get_app()
        _ = app._tr  # Get translation function

        # Do we have unsaved changes?
        if app.project.needs_save():
            ret = QMessageBox.question(
                self,
                _("Unsaved Changes"),
                _("Save changes to project first?"),
                QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                # Save project
                self.actionSave_trigger()
            elif ret == QMessageBox.Cancel:
                # User canceled prompt
                return

        # Clear any previous thumbnails
        self.clear_all_thumbnails()

        # clear data and start new project
        app.project.load("")
        app.updates.reset()
        self.updateStatusChanged(False, False)

        # Reset selections
        self.clearSelections()

        # Refresh files views
        self.refreshFilesSignal.emit()
        log.info("New Project created.")

        # Set Window title
        self.SetWindowTitle()

        # Seek to frame 0
        self.SeekSignal.emit(1)

    def actionAnimatedTitle_trigger(self):
        # show dialog
        from windows.animated_title import AnimatedTitle
        win = AnimatedTitle()
        # Run the dialog event loop - blocking interaction on this window during that time
        result = win.exec_()
        if result == QDialog.Accepted:
            log.info('animated title add confirmed')
        else:
            log.info('animated title add cancelled')

    def actionAnimation_trigger(self):
        # show dialog
        from windows.animation import Animation
        win = Animation()
        # Run the dialog event loop - blocking interaction on this window during that time
        result = win.exec_()
        if result == QDialog.Accepted:
            log.info('animation confirmed')
        else:
            log.info('animation cancelled')

    def actionTitle_trigger(self):
        # show dialog
        from windows.title_editor import TitleEditor
        win = TitleEditor()
        # Run the dialog event loop - blocking interaction on this window during that time
        win.exec_()

    def actionEditTitle_trigger(self):
        # Loop through selected files (set 1 selected file if more than 1)
        for f in self.selected_files():
            if f.data.get("path").endswith(".svg"):
                file_path = f.data.get("path")
                file_id = f.id
                break

        if not file_path:
            return

        # show dialog for editing title
        from windows.title_editor import TitleEditor
        win = TitleEditor(edit_file_path=file_path)
        # Run the dialog event loop - blocking interaction on this window during that time
        win.exec_()

        # Update file thumbnail
        self.FileUpdated.emit(file_id)

        # Force update of clips
        for c in Clip.filter(file_id=file_id):
            # update clip
            c.data["reader"]["path"] = file_path
            c.save()

            # Emit thumbnail update signal (to update timeline thumb image)
            self.ThumbnailUpdated.emit(c.id)

        # Update preview
        self.refreshFrameSignal.emit()

    def actionDuplicateTitle_trigger(self):

        file_path = None

        # Loop through selected files (set 1 selected file if more than 1)
        for f in self.selected_files():
            if f.data.get("path").endswith(".svg"):
                file_path = f.data.get("path")
                break

        if not file_path:
            return

        # show dialog for editing title
        from windows.title_editor import TitleEditor
        win = TitleEditor(edit_file_path=file_path, duplicate=True)
        # Run the dialog event loop - blocking interaction on this window during that time
        return win.exec_()

    def actionClearHistory_trigger(self):
        """Clear history for current project"""
        project = get_app().project
        project.has_unsaved_changes = True
        get_app().updates.reset()
        log.info('History cleared')

    def save_project(self, file_path):
        """ Save a project to a file path, and refresh the screen """
        app = get_app()
        _ = app._tr  # Get translation function

        try:
            # Update history in project data
            s = app.get_settings()
            app.updates.save_history(app.project, s.get("history-limit"))

            # Save project to file
            app.project.save(file_path)

            # Set Window title
            self.SetWindowTitle()

            # Load recent projects again
            self.load_recent_menu()

            log.info("Saved project {}".format(file_path))

        except Exception as ex:
            log.error("Couldn't save project %s.", file_path, exc_info=1)
            QMessageBox.warning(self, _("Error Saving Project"), str(ex))

    def open_project(self, file_path, clear_thumbnails=True):
        """ Open a project from a file path, and refresh the screen """

        app = get_app()
        _ = app._tr  # Get translation function

        # First check for empty file_path (probably user cancellation)
        if not file_path:
            # Ignore the request
            return

        # Stop preview thread
        self.SpeedSignal.emit(0)
        ui_util.setup_icon(self, self.actionPlay, "actionPlay", "media-playback-start")
        self.actionPlay.setChecked(False)
        QCoreApplication.processEvents()

        # Do we have unsaved changes?
        if app.project.needs_save():
            ret = QMessageBox.question(
                self,
                _("Unsaved Changes"),
                _("Save changes to project first?"),
                QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                # Save project
                self.actionSave.trigger()
            elif ret == QMessageBox.Cancel:
                # User canceled prompt
                return

        # Set cursor to waiting
        app.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:
            if os.path.exists(file_path):
                # Clear any previous thumbnails
                if clear_thumbnails:
                    self.clear_all_thumbnails()

                # Load project file
                app.project.load(file_path, clear_thumbnails)

                # Set Window title
                self.SetWindowTitle()

                # Reset undo/redo history
                app.updates.reset()
                app.updates.load_history(app.project)

                # Reset selections
                self.clearSelections()

                # Refresh files views
                self.refreshFilesSignal.emit()

                # Refresh thumbnail
                self.refreshFrameSignal.emit()

                # Load recent projects again
                self.load_recent_menu()

                log.info("Loaded project {}".format(file_path))
            else:
                log.info("File not found at {}".format(file_path))
                self.statusBar.showMessage(
                    _("Project %s is missing (it may have been moved or deleted). "
                      "It has been removed from the Recent Projects menu." % file_path),
                    5000)
                self.remove_recent_project(file_path)
                self.load_recent_menu()

        except Exception as ex:
            log.error("Couldn't open project %s.", file_path, exc_info=1)
            QMessageBox.warning(self, _("Error Opening Project"), str(ex))

        # Restore normal cursor
        app.restoreOverrideCursor()

    def clear_all_thumbnails(self):
        """Clear all user thumbnails"""
        try:
            clear_path = os.path.join(info.USER_PATH, "thumbnail")
            if os.path.exists(clear_path):
                log.info("Clear all thumbnails: %s", clear_path)
                shutil.rmtree(clear_path)
                os.mkdir(clear_path)

            # Clear any blender animations
            clear_path = os.path.join(info.USER_PATH, "blender")
            if os.path.exists(clear_path):
                log.info("Clear all animations: %s", clear_path)
                shutil.rmtree(clear_path)
                os.mkdir(clear_path)

            # Clear any title animations
            clear_path = os.path.join(info.USER_PATH, "title")
            if os.path.exists(clear_path):
                log.info("Clear all titles: %s", clear_path)
                shutil.rmtree(clear_path)
                os.mkdir(clear_path)

            # Clear any backups
            if os.path.exists(info.BACKUP_FILE):
                log.info("Clear backup: %s", info.BACKUP_FILE)
                # Remove backup file
                os.unlink(info.BACKUP_FILE)

        except Exception:
            log.info("Failed to clear %s", clear_path, exc_info=1)

    def actionOpen_trigger(self):
        app = get_app()
        _ = app._tr
        recommended_path = app.project.current_filepath
        if not recommended_path:
            recommended_path = info.HOME_PATH

        # Do we have unsaved changes?
        if app.project.needs_save():
            ret = QMessageBox.question(
                self,
                _("Unsaved Changes"),
                _("Save changes to project first?"),
                QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                # Save project
                self.actionSave_trigger()
            elif ret == QMessageBox.Cancel:
                # User canceled prompt
                return

        # Prompt for open project file
        file_path = QFileDialog.getOpenFileName(
            self,
            _("Open Project..."),
            recommended_path,
            _("OpenShot Project (*.osp)"))[0]

        # Load project file
        self.OpenProjectSignal.emit(file_path)

    def actionSave_trigger(self):
        app = get_app()
        _ = app._tr

        # Get current filepath if any, otherwise ask user
        file_path = app.project.current_filepath
        if not file_path:
            recommended_path = os.path.join(info.HOME_PATH, "%s.osp" % _("Untitled Project"))
            file_path = QFileDialog.getSaveFileName(
                self,
                _("Save Project..."),
                recommended_path,
                _("OpenShot Project (*.osp)"))[0]

        if file_path:
            # Append .osp if needed
            if ".osp" not in file_path:
                file_path = "%s.osp" % file_path

            # Save project
            self.save_project(file_path)

    def auto_save_project(self):
        """Auto save the project"""
        import time

        app = get_app()
        s = app.get_settings()

        # Get current filepath (if any)
        file_path = app.project.current_filepath
        if app.project.needs_save():
            log.info("auto_save_project")

            if file_path:
                # A Real project file exists
                # Append .osp if needed
                if ".osp" not in file_path:
                    file_path = "%s.osp" % file_path
                folder_path, file_name = os.path.split(file_path)
                file_name, file_ext = os.path.splitext(file_name)

                # Make copy of unsaved project file in 'recovery' folder
                recover_path_with_timestamp = os.path.join(
                    info.RECOVERY_PATH, "%d-%s.osp" % (int(time.time()), file_name))
                shutil.copy(file_path, recover_path_with_timestamp)

                # Find any recovery file older than X auto-saves
                old_backup_files = []
                backup_file_count = 0
                for backup_filename in reversed(sorted(os.listdir(info.RECOVERY_PATH))):
                    if ".osp" in backup_filename:
                        backup_file_count += 1
                        if backup_file_count > s.get("recovery-limit"):
                            old_backup_files.append(os.path.join(info.RECOVERY_PATH, backup_filename))

                # Delete recovery files which are 'too old'
                for backup_filepath in old_backup_files:
                    os.unlink(backup_filepath)

                # Save project
                log.info("Auto save project file: %s", file_path)
                self.save_project(file_path)

                # Remove backup.osp (if any)
                if os.path.exists(info.BACKUP_FILE):
                    # Delete backup.osp since we just saved the actual project
                    os.unlink(info.BACKUP_FILE)

            else:
                # No saved project found
                log.info("Creating backup of project file: %s", info.BACKUP_FILE)
                app.project.save(info.BACKUP_FILE, move_temp_files=False, make_paths_relative=False)

                # Clear the file_path (which is set by saving the project)
                app.project.current_filepath = None
                app.project.has_unsaved_changes = True

    def actionSaveAs_trigger(self):
        app = get_app()
        _ = app._tr

        recommended_path = app.project.current_filepath
        if not recommended_path:
            recommended_path = os.path.join(
                info.HOME_PATH, "%s.osp" % _("Untitled Project"))
        file_path = QFileDialog.getSaveFileName(
            self,
            _("Save Project As..."),
            recommended_path,
            _("OpenShot Project (*.osp)"))[0]
        if file_path:
            # Append .osp if needed
            if ".osp" not in file_path:
                file_path = "%s.osp" % file_path

            # Save new project
            self.save_project(file_path)

    ### CHANGED ###

    def actionClearText_trigger(self):
        app = get_app()
        _ = app._tr

        self.textTextEdit.clear()
        self.filepath = None

    def actionSaveText_trigger(self):
        app = get_app()
        _ = app._tr

        if self.filepath is not None:
            text_in_editor = self.textTextEdit.toPlainText()
            f = open(self.filepath, 'w')
            f.write(text_in_editor)
            f.close()
        else:
            print("Please open some transcript")

    def actionClearTextSlide_trigger(self):
        app = get_app()
        _ = app._tr

        self.slideTextEdit.clear()

    # def actionSaveTextSlide_trigger(self):
    #     app = get_app()
    #     _ = app._tr

    #     text_in_editor = self.slideTextEdit.toPlainText()
    #     print(text_in_editor)

    def actionFOMM_trigger(self):
        log.info("actionFOMM_trigger")
        # text_in_editor = self.slideTextEdit.toPlainText()
        name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
        img_pth = os.path.join(sys.path[0],'exports', name + '.png')
        openshot.Timeline.GetFrame(
            self.timeline_sync.timeline, self.preview_thread.current_frame).Save(img_pth, 1.0)
        try:
            self.progressFOMM = QProgressDialog('Performing FOMM', '', 0, 1, self)
            self.progressFOMM.setWindowTitle("Work in progress")
            self.progressFOMM.setWindowModality(Qt.WindowModal)
            self.progressFOMM.show()
            self.progressFOMM.setValue(0)
            self.doGenerateFOMM(img_pth)
        except:
            errBox = QMessageBox()
            errBox.setWindowTitle('Error')
            errBox.setText('Error: ' + str(e))
            errBox.addButton(QMessageBox.Ok)
            errBox.exec()
            return

    def doGenerateFOMM(self, pth):
        self.runnableFOMM = FOMM(self.findAllMarkerPositions(), self.url_fomm, self.progressFOMM, pth)
        QThreadPool.globalInstance().start(self.runnableFOMM)

    class MIT_thread(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(str)

        def __init__(self, pos, url_mit, pth):
            QObject.__init__(self)
            self.pos = pos
            self.url_mit = url_mit
            self.pth = pth

        def run(self):

            try:
                name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
                ls_pos = sorted(self.pos['ls'])
                
                fps = get_app().project.get("fps")
                fps_float = float(fps["num"]) / float(fps["den"])

                ls_pos = [int(fps_float * x + 1) for x in ls_pos[1:-1]]
                # print(ls_pos)
                from windows.export import Export
                win = Export()
                print(ls_pos, name)
                print(ls_pos)
                fn, epn = win.accept(ls_pos[0], ls_pos[1], True, name)
                print(ls_pos, name, fn, epn)

                outdir = os.path.join(sys.path[0], 'ls_out', 'to_send')
                pth2 = os.path.join(outdir, name + '.wav')
                audio_cmd = 'ffmpeg -i {} {}'.format(epn, pth2)

                subprocess.call(audio_cmd, shell=True)

                outfile = os.path.join(sys.path[0], 'ls_out', 'res', name + '.mp4')

                cmd = 'curl -X POST -F file1=@{} -F file2=@{} {} > {}'.format(self.pth, pth2, self.url_mit, outfile)
                print(cmd)
                subprocess.call(cmd, shell=True)
            except:
                outfile = "None"
                print("Faced some error in TTS thread")        
            # sleep(5);
            self.progress.emit(outfile)
            self.finished.emit()

    def actionMakeItTalk_trigger(self):
        log.info("actionMIT_trigger")
          # Translate object
        app = get_app()
        _ = app._tr

        
        # text_in_editor = self.slideTextEdit.toPlainText()
        name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
        img_pth = os.path.join(sys.path[0], 'exports', name + '.png')
        # Set MaxSize to full project resolution and clear preview cache so we get a full resolution frame
        self.timeline_sync.timeline.SetMaxSize(app.project.get("width"), app.project.get("height"))
        
        #self.cache_object.Clear()

           # Check if file exists, if it does, get the lastModified time
        if os.path.exists(img_pth):
            imagePathTime = QFileInfo(img_pth).lastModified()
        else:
            imagePathTime = QDateTime()

        openshot.Timeline.GetFrame(
            self.timeline_sync.timeline, self.preview_thread.current_frame).Save(img_pth, 1.0)
        

        # Show message to user
        if os.path.exists(img_pth) and (QFileInfo(img_pth).lastModified() > imagePathTime):
            print("Saved Frame to %s" % img_pth)
        else:
            print("Failed to save image to %s" % img_pth)

        
        #print(img_pth)

        self.MITthread = QThread()
        self.MITworker = self.MIT_thread(self.findAllMarkerPositions(), self.url_mit, img_pth);
        self.MITworker.moveToThread(self.MITthread)
        self.MITthread.started.connect(self.MITworker.run)
        self.MITworker.finished.connect(self.MITthread.quit)
        self.MITworker.finished.connect(self.MITworker.deleteLater)
        self.MITthread.finished.connect(self.MITthread.deleteLater)
        self.MITworker.progress.connect(self.MITfinished)

        self.MITthread.start()
        self.MITMsgBox = QMessageBox()
        self.MITMsgBox.setWindowTitle('Work in progess')
        self.MITMsgBox.setText('Running Make It Talk...')
        # self.MITMsgBox.addButton(QMessageBox.Ok)
        self.MITMsgBox.show()

        self.MITthread.finished.connect(
            lambda: self.MITMsgBox.hide()
        )
    def MITfinished(self, filename):
        qfilename = QUrl.fromLocalFile(filename)
        self.files_model.process_urls([qfilename]) 

    def actionFiller_trigger(self):
        print("in actionFiller_trigger")
        app = get_app()
        _ = app._tr
        try:
            name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
            filename1 = os.path.join(sys.path[0], "filler_out", 'filler_' + name + '.mp4')
            filename2 = os.path.join(sys.path[0], "filler_out", name + ".jpg")

            qfilename1 = QUrl.fromLocalFile(filename1)
            qfilename2 = QUrl.fromLocalFile(filename2)

            cmd1 = 'curl -X POST -F file=@{} {}filler/{} > {}'.format(self.video_filename, self.url, name, filename1)
            subprocess.call(cmd1, shell=True)

            cmd2 = 'curl {}filler_image/{}.jpg > {}'.format(self.url, name, filename2)
            subprocess.call(cmd2, shell=True)
            self.files_model.process_urls([qfilename1, qfilename2])
        except Exception as e:
                log.info(e)
                print("Some error in filler")

    class SlideTranslate2_thread(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(str)

        def __init__(self, text, url_sp2, filename_exported):
            QObject.__init__(self)
            self.text = text
            self.url_sp2 = url_sp2
            self.filename_exported = filename_exported

        def run(self):

            log.info("Worker actionSlideTranslate2_trigger")
            try:
                name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
                fps = get_app().project.get("fps")
                fps_float = float(fps["num"]) / float(fps["den"])
                
                outfile = os.path.join(sys.path[0], 'ls_out', 'res', name + '.avi')
                txtfile = os.path.join(sys.path[0], 'ls_out', 'res', name + '.txt')
                with open(txtfile, 'w', encoding='utf-8') as f:
                    for txt in self.text:
                        f.write(txt)
                f.close()
                cmd1 = 'curl -X POST -F file1=@{} {}/{}/{} > {}'.format(txtfile, self.url_sp2, self.filename_exported, int(fps_float), outfile)
                subprocess.call(cmd1, shell=True)
            except:
                txt = ""
                print("Some error in SlideTranslate Thread")
            self.progress.emit(outfile)
            self.finished.emit()

    def actionSaveTextSlide_trigger(self):
        log.info("actionSaveTextSlide2_trigger")
        text_in_editor = self.slideTextEdit.toPlainText()
        
        # self.slide_ocr = slide_class.SlidesOCR()
        self.ST2thread = QThread()
        self.ST2worker = self.SlideTranslate2_thread(text_in_editor, self.url_sp2, self.slide_exported_file);
        self.ST2worker.moveToThread(self.ST2thread)
        self.ST2thread.started.connect(self.ST2worker.run)
        self.ST2worker.finished.connect(self.ST2thread.quit)
        self.ST2worker.finished.connect(self.ST2worker.deleteLater)
        self.ST2thread.finished.connect(self.ST2thread.deleteLater)
        self.ST2worker.progress.connect(self.ST2finished)
        self.ST2thread.start()
        self.ST2MessageBox = QMessageBox()
        self.ST2MessageBox.setWindowTitle('Work in progess')
        self.ST2MessageBox.setText('Overlaying text on slide...')
        # self.ST2MessageBox.addButton(QMessageBox.Ok)
        self.ST2MessageBox.show()

        self.ST2thread.finished.connect(
            lambda: self.ST2MessageBox.hide()
        )

    def ST2finished(self, outfile):
        qfilename = QUrl.fromLocalFile(outfile)
        self.files_model.process_urls([qfilename])

    class SlideTranslate_thread(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(str)

        def __init__(self, pos, url_sp1):
            QObject.__init__(self)
            self.pos = pos
            self.url_sp1 = url_sp1

        def run(self):

            log.info("Worker actionSlideTranslate1_trigger")
            try:
                name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
                ls_pos = sorted(self.pos['ls'])
                fps = get_app().project.get("fps")
                fps_float = float(fps["num"]) / float(fps["den"])
                ls_pos = [int(fps_float * x + 1) for x in ls_pos[1:-1]]

                from windows.export import Export
                win = Export()
                fn, epn = win.accept(ls_pos[0], ls_pos[1], True, name)
                
                outfile = os.path.join(sys.path[0], 'ls_out', 'res', name + '.txt')
                cmd1 = 'curl -X POST -F file1=@{} {} > {}'.format(epn, self.url_sp1, outfile)
                print(cmd1)
                subprocess.call(cmd1, shell=True)
            except Exception as e:
                log.info(e)
                outfile=None
                print("Some error in SlideTranslate Thread")
            self.progress.emit(outfile)
            self.finished.emit()


    def actionSlideTranslate_trigger(self):


        # self.slide_ocr = slide_class.SlidesOCR()
        self.thread = QThread()
        self.worker = self.SlideTranslate_thread(self.findAllMarkerPositions(), self.url_sp1);
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.finfin)
        log.info("actionSlideTranslate_trigger")
        self.thread.start()
        self.ST1MessageBox = QMessageBox()
        self.ST1MessageBox.setWindowTitle('Work in progess')
        self.ST1MessageBox.setText('Getting text from slide...')
        # self.ST1MessageBox.addButton(QMessageBox.Ok)
        self.ST1MessageBox.show()

        self.thread.finished.connect(
            lambda: self.ST1MessageBox.hide()
        )


    class SpeechToText_thread(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(str)

        def __init__(self, pos, url_stt):
            QObject.__init__(self)
            self.pos = pos
            self.url_stt = url_stt

        def run(self):

            log.info("Worker actionSpeechToText_trigger")
            try:
                name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
                ls_pos = sorted(self.pos['ls'])
                fps = get_app().project.get("fps")
                fps_float = float(fps["num"]) / float(fps["den"])
                ls_pos = [int(fps_float * x + 1) for x in ls_pos[1:-1]]

                from windows.export import Export
                win = Export()
                fn, epn = win.accept(ls_pos[0], ls_pos[1], True, name)
                outdir = os.path.join(sys.path[0], 'stt_out', 'to_send')
                pth2 = os.path.join(outdir, name + '.wav')
                audio_cmd = 'ffmpeg -i {} {}'.format(epn, pth2)
                print(audio_cmd)
                subprocess.call(audio_cmd, shell=True)
                outfile = os.path.join(sys.path[0], 'stt_out', 'res', name + '.txt')
                cmd1 = 'curl -X POST -F file1=@{} {} > {}'.format(pth2, self.url_stt, outfile)
                subprocess.call(cmd1, shell=True)
            except Exception as e:
                print(e)
                outfile = ""
                print("Some error in SpeechToText Thread")
            self.progress.emit(outfile)
            self.finished.emit()


    def actionSpeechToText_trigger(self):


        # self.slide_ocr = slide_class.SlidesOCR()
        self.thread = QThread()
        self.worker = self.SpeechToText_thread(self.findAllMarkerPositions(), self.url_stt);
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.finfin1)
        log.info("actionSpeechToText_trigger")
        self.thread.start()
        self.STTMessageBox = QMessageBox()
        self.STTMessageBox.setWindowTitle('Work in progess')
        self.STTMessageBox.setText('Getting text from audio...')
        # self.ST1MessageBox.addButton(QMessageBox.Ok)
        self.STTMessageBox.show()

        self.thread.finished.connect(
            lambda: self.STTMessageBox.hide()
        )


    def finfin(self, x):
        print("Finfin", x)
        self.slide_exported_file = x.split('/')[-1].split('.')[0]
        f = open(x, 'r', encoding='utf-8')
        txt = f.readlines()
        # print(txt)
        text = "".join(txt)
        self.slideTextEdit.clear()
        self.slideTextEdit.insertPlainText(text)

    def finfin1(self, x):
        print("Finfin1", x)
        self.slide_exported_file = x.split('/')[-1].split('.')[0]
        f = open(x, 'r', encoding='utf-8')
        txt = f.readlines()
        # print(txt)
        text = "".join(txt)
        self.textTextEdit.clear()
        self.textTextEdit.insertPlainText(text)




    class TTS_thread(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(str)

        def __init__(self, text, url_tts):
            QObject.__init__(self)
            self.text = text
            self.url_tts = url_tts

        def run(self):

            try:
                log.info("In TTS thread")
                name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
                filename_text = os.path.join(sys.path[0], "tts_out", name + '.txt')
                filename = os.path.join(sys.path[0], "tts_out", name + ".wav")
                print(filename,filename_text)
                with open(filename_text, 'w',encoding='utf-8') as f:
                    f.write(self.text)
                f.close()
                #print(filename,filename_text)
                cmd = 'curl -X POST -F file=@{} {} > {}'.format(filename_text, self.url_tts, filename)
                print(cmd)
                subprocess.call(cmd, shell=True)
            except:
                filename = "None"
                print("Faced some error in TTS thread")        
            # sleep(5);
            self.progress.emit(filename)
            self.finished.emit()


    class TextTranslate_thread(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(str)

        def __init__(self, text, url_trans,lang):
            QObject.__init__(self)
            self.text = text
            self.url_trans = url_trans
            self.lang = lang

        def run(self):

            try:
                log.info("In Translate thread")
                name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
                filename_text = os.path.join(sys.path[0], "tts_out", name + '.txt')
                #filename = os.path.join(sys.path[0], "tts_out", name + ".txt")
                #print(filename,filename_text)
                #print("fine")
                with open(filename_text, 'w',encoding='utf8') as f:
                    #print("fine")
                    f.write(self.text)
                    #print("fine")
                f.close()
                print(self.text,"3")
                #print(filename,filename_text)
                translated= filename_text.replace(".txt","a.txt")
                cmd = 'curl -X POST -F file=@{} {}/{} > {}'.format(filename_text, self.url_trans,self.lang ,translated)
                print(cmd)
                subprocess.call(cmd, shell=True)
            except Exception as e:
                log.info(e)
                translated = "None"
                print("Faced some error in Translate thread")        
            # sleep(5);
            self.progress.emit(translated)
            self.finished.emit()

    def actiontextTranslate_trigger(self,q):
        #languages={'bn': 'Bengali','en':"English",'hi': 'Hindi','ml': 'Malayalam','mr': 'Marathi','ta': 'Tamil','te': 'Telugu'}
        langs={'Bengali':'bn',"English":"en",'Hindi':"hi",'Malayalam':"ml",'Marathi':"mr",'Tamil':'ta','Telugu':'te'}
        log.info("Translate_trigger")
        self.Translatethread = QThread()
        print(self.textTextEdit.toPlainText())
        self.Translateworker = self.TextTranslate_thread(self.textTextEdit.toPlainText(), self.url_trans,langs[q.text()]);
        self.Translateworker.moveToThread(self.Translatethread)
        self.Translatethread.started.connect(self.Translateworker.run)
        self.Translateworker.finished.connect(self.Translatethread.quit)
        self.Translateworker.finished.connect(self.Translateworker.deleteLater)
        self.Translatethread.finished.connect(self.Translatethread.deleteLater)
        self.Translateworker.progress.connect(self.finfin1)
        self.Translatethread.start()
        self.TranslateMsgBox = QMessageBox()
        self.TranslateMsgBox.setWindowTitle('Work in progess')
        self.TranslateMsgBox.setText('Translating to {}...'.format(q.text()))
        # self.TTSMsgBox.addButton(QMessageBox.Ok)
        self.TranslateMsgBox.show()

        self.Translatethread.finished.connect(
            lambda: self.TranslateMsgBox.hide()
        )

    def actionGlow_trigger(self):
        log.info("actionGlow_trigger")

        self.TTSthread = QThread()
        self.TTSworker = self.TTS_thread(self.textTextEdit.toPlainText(), self.url_tts);
        self.TTSworker.moveToThread(self.TTSthread)
        self.TTSthread.started.connect(self.TTSworker.run)
        self.TTSworker.finished.connect(self.TTSthread.quit)
        self.TTSworker.finished.connect(self.TTSworker.deleteLater)
        self.TTSthread.finished.connect(self.TTSthread.deleteLater)
        self.TTSworker.progress.connect(self.TTSfinished)

        self.TTSthread.start()
        self.TTSMsgBox = QMessageBox()
        self.TTSMsgBox.setWindowTitle('Work in progess')
        self.TTSMsgBox.setText('Getting audio for the text...')
        # self.TTSMsgBox.addButton(QMessageBox.Ok)
        self.TTSMsgBox.show()

        self.TTSthread.finished.connect(
            lambda: self.TTSMsgBox.hide()
        )
    def TTSfinished(self, filename):
        qfilename = QUrl.fromLocalFile(filename)
        self.files_model.process_urls([qfilename])        

    def actionRTVC_trigger(self):
        app = get_app()
        _ = app._tr

        print("HERE IN RTVC FUNCTION", self.textTextEdit.toPlainText())
    

    def actionGoogleTSS_trigger(self):
        app = get_app()
        _ = app._tr

        print("HERE IN GTTS FUNCTION", self.textTextEdit.toPlainText())
        self.TTSthread = QThread()
        self.TTSworker = self.TTS_thread(self.textTextEdit.toPlainText(), self.url_tts2);
        self.TTSworker.moveToThread(self.TTSthread)
        self.TTSthread.started.connect(self.TTSworker.run)
        self.TTSworker.finished.connect(self.TTSthread.quit)
        self.TTSworker.finished.connect(self.TTSworker.deleteLater)
        self.TTSthread.finished.connect(self.TTSthread.deleteLater)
        self.TTSworker.progress.connect(self.TTSfinished)

        self.TTSthread.start()
        self.TTSMsgBox = QMessageBox()
        self.TTSMsgBox.setWindowTitle('Work in progess')
        self.TTSMsgBox.setText('Getting audio for the text...')
        # self.TTSMsgBox.addButton(QMessageBox.Ok)
        self.TTSMsgBox.show()

        self.TTSthread.finished.connect(
            lambda: self.TTSMsgBox.hide()
        )

    
    def actionImportTextFiles_trigger(self):
        app = get_app()
        _ = app._tr

        recommended_path = app.project.get("import_path")
        if not recommended_path or not os.path.exists(recommended_path):
            recommended_path = os.path.join(info.HOME_PATH)

        # PyQt through 5.13.0 had the 'directory' argument mis-typed as str
        if PYQT_VERSION_STR < '5.13.1':
            dir_type = "str"
            start_location = str(recommended_path)
        else:
            dir_type = "QUrl"
            start_location = QUrl.fromLocalFile(recommended_path)

        log.debug("Calling getOpenFileURLs() with %s directory argument", dir_type)
        qurl_list = QFileDialog.getOpenFileUrls(
            self,
            _("Import Files..."),
            start_location,
            )[0]

        # Set cursor to waiting
        app.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:
            # Import list of files
            # self.files_model.process_urls(qurl_list)
            filepath = qurl_list[0].toLocalFile()
            self.filepath = filepath
            print(filepath)
            text = open(filepath, 'r').readlines()[0]
            # print(text)

            # # Get cursor / current line of text (where cursor is located)
            # cursor = self.textTextEdit.textCursor()
            # self.textTextEdit.moveCursor(QTextCursor.StartOfLine)
            # line_text = cursor.block().text()
            # self.textTextEdit.moveCursor(QTextCursor.EndOfLine)
            self.textTextEdit.clear()
            self.textTextEdit.insertPlainText(text)

            # Refresh files views
            self.refreshFilesSignal.emit()
        finally:
            # Restore cursor
            app.restoreOverrideCursor()

    ### CHANGED ###

    def actionImportFiles_trigger(self):
        app = get_app()
        _ = app._tr

        recommended_path = app.project.get("import_path")
        if not recommended_path or not os.path.exists(recommended_path):
            recommended_path = os.path.join(info.HOME_PATH)

        # PyQt through 5.13.0 had the 'directory' argument mis-typed as str
        if PYQT_VERSION_STR < '5.13.1':
            dir_type = "str"
            start_location = str(recommended_path)
        else:
            dir_type = "QUrl"
            start_location = QUrl.fromLocalFile(recommended_path)

        log.debug("Calling getOpenFileURLs() with %s directory argument", dir_type)
        qurl_list = QFileDialog.getOpenFileUrls(
            self,
            _("Import Files..."),
            start_location,
            )[0]

        # print(qurl_list[0])
        self.video_filename = qurl_list[0].toLocalFile()
        # Set cursor to waiting
        app.setOverrideCursor(QCursor(Qt.WaitCursor))

        try:
            # Import list of files
            self.files_model.process_urls(qurl_list)

            # Refresh files views
            self.refreshFilesSignal.emit()
        finally:
            # Restore cursor
            app.restoreOverrideCursor()

    def invalidImage(self, filename=None):
        """ Show a popup when an image file can't be loaded """
        if not filename:
            return

        # Translations
        _ = get_app()._tr

        # Show message to user
        QMessageBox.warning(
            self,
            None,
            _("%s is not a valid video, audio, or image file.") % filename,
            QMessageBox.Ok
        )

    def promptImageSequence(self, filename=None):
        """ Ask the user whether to import an image sequence """
        if not filename:
            return False

        # Get translations
        app = get_app()
        _ = app._tr

        # Process the event queue first, since we've been ignoring input
        app.processEvents()

        # Display prompt dialog
        ret = QMessageBox.question(
            self,
            _("Import Image Sequence"),
            _("Would you like to import %s as an image sequence?") % filename,
            QMessageBox.No | QMessageBox.Yes
        )
        return bool(ret == QMessageBox.Yes)

    def actionAdd_to_Timeline_trigger(self, checked=False):
        # Loop through selected files
        files = self.selected_files()

        # Bail if nothing's selected
        if not files:
            return

        # Get current position of playhead
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])
        pos = (self.preview_thread.player.Position() - 1) / fps_float

        # show window
        from windows.add_to_timeline import AddToTimeline
        win = AddToTimeline(files, pos)
        # Run the dialog event loop - blocking interaction on this window during this time
        result = win.exec_()
        if result == QDialog.Accepted:
            log.info('confirmed')
        else:
            log.info('canceled')

    def actionExportVideo_trigger(self, checked=True):
        # show window
        from windows.export import Export
        win = Export()
        # Run the dialog event loop - blocking interaction on this window during this time
        result = win.exec_()
        if result == QDialog.Accepted:
            log.info('Export Video add confirmed')
        else:
            log.info('Export Video add cancelled')

    def actionExportEDL_trigger(self, checked=True):
        """Export EDL File"""
        export_edl()

    def actionExportFCPXML_trigger(self, checked=True):
        """Export XML (Final Cut Pro) File"""
        export_xml()

    def actionImportEDL_trigger(self, checked=True):
        """Import EDL File"""
        import_edl()

    def actionImportFCPXML_trigger(self, checked=True):
        """Import XML (Final Cut Pro) File"""
        import_xml()

    def actionUndo_trigger(self, checked=True):
        log.info('actionUndo_trigger')
        get_app().updates.undo()

        # Update the preview
        self.refreshFrameSignal.emit()

    def actionRedo_trigger(self, checked=True):
        log.info('actionRedo_trigger')
        get_app().updates.redo()

        # Update the preview
        self.refreshFrameSignal.emit()

    def actionPreferences_trigger(self, checked=True):
        # Stop preview thread
        self.SpeedSignal.emit(0)
        ui_util.setup_icon(self, self.actionPlay, "actionPlay", "media-playback-start")
        self.actionPlay.setChecked(False)

        # Set cursor to waiting
        get_app().setOverrideCursor(QCursor(Qt.WaitCursor))

        # Show dialog
        from windows.preferences import Preferences
        win = Preferences()
        # Run the dialog event loop - blocking interaction on this window during this time
        result = win.exec_()
        if result == QDialog.Accepted:
            log.info('Preferences add confirmed')
        else:
            log.info('Preferences add cancelled')

        # Save settings
        s = get_app().get_settings()
        s.save()

        # Restore normal cursor
        get_app().restoreOverrideCursor()

    def actionFilesShowAll_trigger(self, checked=True):
        self.refreshFilesSignal.emit()

    def actionFilesShowVideo_trigger(self, checked=True):
        self.refreshFilesSignal.emit()

    def actionFilesShowAudio_trigger(self, checked=True):
        self.refreshFilesSignal.emit()

    def actionFilesShowImage_trigger(self, checked=True):
        self.refreshFilesSignal.emit()

    def actionTransitionsShowAll_trigger(self, checked=True):
        self.refreshTransitionsSignal.emit()

    def actionTransitionsShowCommon_trigger(self, checked=True):
        self.refreshTransitionsSignal.emit()

    def actionHelpContents_trigger(self, checked=True):
        try:
            webbrowser.open("https://www.openshot.org/%suser-guide/?app-menu" % info.website_language(), new=1)
        except Exception:
            QMessageBox.information(self, "Error !", "Unable to open the online help")
            log.error("Unable to open the Help Contents", exc_info=1)

    def actionAbout_trigger(self, checked=True):
        """Show about dialog"""
        from windows.about import About
        win = About()
        # Run the dialog event loop - blocking interaction on this window during this time
        win.exec_()

    def actionReportBug_trigger(self, checked=True):
        try:
            webbrowser.open("https://www.openshot.org/%sissues/new/?app-menu" % info.website_language(), new=1)
        except Exception:
            QMessageBox.information(self, "Error !", "Unable to open the Bug Report GitHub Issues web page")
            log.error("Unable to open the Bug Report page", exc_info=1)

    def actionAskQuestion_trigger(self, checked=True):
        try:
            webbrowser.open("https://www.reddit.com/r/OpenShot/", new=1)
        except Exception:
            QMessageBox.information(self, "Error !", "Unable to open the official OpenShot subreddit web page")
            log.error("Unable to open the subreddit page", exc_info=1)

    def actionTranslate_trigger(self, checked=True):
        try:
            webbrowser.open("https://translations.launchpad.net/openshot/2.0", new=1)
        except Exception:
            QMessageBox.information(self, "Error !", "Unable to open the Translation web page")
            log.error("Unable to open the translation page", exc_info=1)

    def actionDonate_trigger(self, checked=True):
        try:
            webbrowser.open("https://www.openshot.org/%sdonate/?app-menu" % info.website_language(), new=1)
        except Exception:
            QMessageBox.information(self, "Error !", "Unable to open the Donate web page")
            log.error("Unable to open the donation page", exc_info=1)

    def actionUpdate_trigger(self, checked=True):
        try:
            webbrowser.open("https://www.openshot.org/%sdownload/?app-toolbar" % info.website_language(), new=1)
        except Exception:
            QMessageBox.information(self, "Error !", "Unable to open the Download web page")
            log.error("Unable to open the download page", exc_info=1)

    def actionPlay_trigger(self, checked=True, force=None):
        if force == "pause":
            self.actionPlay.setChecked(False)
        elif force == "play":
            self.actionPlay.setChecked(True)

        if self.actionPlay.isChecked():
            # Determine max frame (based on clips)
            max_frame = get_app().window.timeline_sync.timeline.GetMaxFrame()
            ui_util.setup_icon(self, self.actionPlay, "actionPlay", "media-playback-pause")
            self.PlaySignal.emit(max_frame)

        else:
            ui_util.setup_icon(self, self.actionPlay, "actionPlay")  # to default
            self.PauseSignal.emit()

    def actionPreview_File_trigger(self, checked=True):
        """ Preview the selected media file """
        log.info('actionPreview_File_trigger')

        # Loop through selected files (set 1 selected file if more than 1)
        f = self.files_model.current_file()

        # Bail out if no file selected
        if not f:
            log.info("Couldn't find current file for preview window")
            return

        # show dialog
        from windows.cutting import Cutting
        win = Cutting(f, preview=True)
        win.show()

    def previewFrame(self, position_frames):
        """Preview a specific frame"""
        # Notify preview thread
        self.previewFrameSignal.emit(position_frames)

        # Notify properties dialog
        self.propertyTableView.select_frame(position_frames)

    def handlePausedVideo(self):
        """Handle the pause signal, by refreshing the properties dialog"""
        self.propertyTableView.select_frame(self.preview_thread.player.Position())

    def movePlayhead(self, position_frames):
        """Update playhead position"""
        # Notify preview thread
        self.timeline.movePlayhead(position_frames)

    def SetPlayheadFollow(self, enable_follow):
        """ Enable / Disable follow mode """
        self.timeline.SetPlayheadFollow(enable_follow)

    def actionFastForward_trigger(self, checked=True):

        # Get the video player object
        player = self.preview_thread.player

        if player.Speed() + 1 != 0:
            self.SpeedSignal.emit(player.Speed() + 1)
        else:
            self.SpeedSignal.emit(player.Speed() + 2)

        if player.Mode() == openshot.PLAYBACK_PAUSED:
            self.actionPlay.trigger()

    def actionRewind_trigger(self, checked=True):
        # Get the video player object
        player = self.preview_thread.player

        new_speed = player.Speed() - 1
        if new_speed == 0:
            new_speed -= 1
        log.debug("Setting speed to %s", new_speed)

        if player.Mode() == openshot.PLAYBACK_PAUSED:
            self.actionPlay.trigger()
        self.SpeedSignal.emit(new_speed)

    def actionJumpStart_trigger(self, checked=True):
        log.debug("actionJumpStart_trigger")

        # Seek to the 1st frame
        self.SeekSignal.emit(1)

    def actionJumpEnd_trigger(self, checked=True):
        log.debug("actionJumpEnd_trigger")

        # Determine last frame (based on clips) & seek there
        max_frame = get_app().window.timeline_sync.timeline.GetMaxFrame()
        self.SeekSignal.emit(max_frame)

    def actionSaveFrame_trigger(self, checked=True):
        log.info("actionSaveFrame_trigger")

        # Translate object
        app = get_app()
        _ = app._tr

        # Prepare to use the status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Determine path for saved frame - Default export path
        recommended_path = recommended_path = os.path.join(info.HOME_PATH)
        if app.project.current_filepath:
            recommended_path = os.path.dirname(app.project.current_filepath)

        # Determine path for saved frame - Project's export path
        if app.project.get("export_path"):
            recommended_path = app.project.get("export_path")

        framePath = "%s/Frame-%05d.png" % (recommended_path, self.preview_thread.current_frame)

        # Ask user to confirm or update framePath
        framePath = QFileDialog.getSaveFileName(self, _("Save Frame..."), framePath, _("Image files (*.png)"))[0]

        if not framePath:
            # No path specified (save frame cancelled)
            self.statusBar.showMessage(_("Save Frame cancelled..."), 5000)
            return

        # Append .png if needed
        if not framePath.endswith(".png"):
            framePath = "%s.png" % framePath

        app.updates.update_untracked(["export_path"], os.path.dirname(framePath))
        log.info("Saving frame to %s", framePath)

        # Pause playback (to prevent crash since we are fixing to change the timeline's max size)
        self.actionPlay_trigger(force="pause")

        # Save current cache object and create a new CacheMemory object (ignore quality and scale prefs)
        old_cache_object = self.cache_object
        new_cache_object = openshot.CacheMemory(app.get_settings().get("cache-limit-mb") * 1024 * 1024)
        self.timeline_sync.timeline.SetCache(new_cache_object)

        # Set MaxSize to full project resolution and clear preview cache so we get a full resolution frame
        self.timeline_sync.timeline.SetMaxSize(app.project.get("width"), app.project.get("height"))
        self.cache_object.Clear()

        # Check if file exists, if it does, get the lastModified time
        if os.path.exists(framePath):
            framePathTime = QFileInfo(framePath).lastModified()
        else:
            framePathTime = QDateTime()

        # Get and Save the frame
        # (return is void, so we cannot check for success/fail here
        # - must use file modification timestamp)
        openshot.Timeline.GetFrame(
            self.timeline_sync.timeline, self.preview_thread.current_frame).Save(framePath, 1.0)
        print(framePath,"HMM")
        # Show message to user
        if os.path.exists(framePath) and (QFileInfo(framePath).lastModified() > framePathTime):
            self.statusBar.showMessage(_("Saved Frame to %s" % framePath), 5000)
        else:
            self.statusBar.showMessage(_("Failed to save image to %s" % framePath), 5000)

        # Reset the MaxSize to match the preview and reset the preview cache
        viewport_rect = self.videoPreview.centeredViewport(self.videoPreview.width(), self.videoPreview.height())
        self.timeline_sync.timeline.SetMaxSize(viewport_rect.width(), viewport_rect.height())
        self.cache_object.Clear()
        self.timeline_sync.timeline.SetCache(old_cache_object)
        self.cache_object = old_cache_object
        old_cache_object = None
        new_cache_object = None

    def renumber_all_layers(self, insert_at=None, stride=1000000):
        """Renumber all of the project's layers to be equidistant (in
        increments of stride), leaving room for future insertion/reordering.
        Inserts a new track, if passed an insert_at index"""

        app = get_app()

        # Don't track renumbering in undo history
        app.updates.ignore_history = True

        tracks = sorted(app.project.get("layers"), key=lambda x: x['number'])

        log.warning("######## RENUMBERING TRACKS ########")
        log.info("Tracks before: {}".format([{x['number']: x['id']} for x in reversed(tracks)]))

        # Leave placeholder for new track, if insert requested
        if insert_at is not None and int(insert_at) < len(tracks) + 1:
            tracks.insert(int(insert_at), "__gap__")

        # Statistics for end-of-function logging
        renum_count = len(tracks)
        renum_min = stride
        renum_max = renum_count * stride

        # Collect items to renumber
        targets = []
        for (idx, layer) in enumerate(tracks):
            newnum = (idx + 1) * stride

            # Check for insertion placeholder
            if isinstance(layer, str) and layer == "__gap__":
                insert_num = newnum
                continue

            # Look up track info
            oldnum = layer.get('number')
            cur_track = Track.get(number=oldnum)
            if not cur_track:
                log.error('Track number %s not found', oldnum)
                continue

            # Find track elements
            cur_clips = list(Clip.filter(layer=oldnum))
            cur_trans = list(Transition.filter(layer=oldnum))

            # Collect items to be updated with new layer number
            targets.append({
                "number": newnum,
                "track": cur_track,
                "clips": cur_clips,
                "trans": cur_trans,
            })

        # Renumber everything
        for layer in targets:
            try:
                num = layer["number"]
                layer["track"].data["number"] = num
                layer["track"].save()

                for item in layer["clips"] + layer["trans"]:
                    item.data["layer"] = num
                    item.save()
            except (AttributeError, IndexError, ValueError):
                # Ignore references to deleted objects
                continue

        # Re-enable undo tracking for new track insertion
        app.updates.ignore_history = False

        # Create new track and insert at gap point, if requested
        if insert_at is not None:
            track = Track()
            track.data = {"number": insert_num, "y": 0, "label": "", "lock": False}
            track.save()

        log.info("Renumbered {} tracks from {} to {}{}".format(
            renum_count, renum_min, renum_max,
            " (inserted {} at {})".format(insert_num, insert_at) if insert_at else "")
        )

    def actionAddTrack_trigger(self, checked=True):
        log.info("actionAddTrack_trigger")

        # Get # of tracks
        all_tracks = get_app().project.get("layers")
        all_tracks.sort(key=lambda x: x['number'], reverse=True)
        track_number = all_tracks[0].get("number") + 1000000

        # Create new track above existing layer(s)
        track = Track()
        track.data = {"number": track_number, "y": 0, "label": "", "lock": False}
        track.save()

    def actionAddTrackAbove_trigger(self, checked=True):
        # Get selected track
        all_tracks = get_app().project.get("layers")
        selected_layer_id = self.selected_tracks[0]

        log.info("adding track above %s", selected_layer_id)

        # Get track data for selected track
        existing_track = Track.get(id=selected_layer_id)
        if not existing_track:
            # Log error and fail silently
            log.error('No track object found with id: %s', selected_layer_id)
            return
        selected_layer_num = int(existing_track.data["number"])

        # Find track above selected track (if any)
        try:
            tracks = sorted(all_tracks, key=lambda x: x['number'])
            existing_index = tracks.index(existing_track.data)
        except ValueError:
            log.warning("Could not find track %s", selected_layer_num, exc_info=1)
            return
        try:
            next_index = existing_index + 1
            next_layer = tracks[next_index]
            delta = abs(selected_layer_num - next_layer.get('number'))
        except IndexError:
            delta = 2000000

        # Calculate new track number (based on gap delta)
        if delta > 2:
            # New track number (pick mid point in track number gap)
            new_track_num = selected_layer_num + int(round(delta / 2.0))

            # Create new track and insert
            track = Track()
            track.data = {"number": new_track_num, "y": 0, "label": "", "lock": False}
            track.save()
        else:
            # Track numbering is too tight, renumber them all and insert
            self.renumber_all_layers(insert_at=next_index)

        tracks = sorted(get_app().project.get("layers"), key=lambda x: x['number'])

        # Temporarily for debugging
        log.info("Tracks after: {}".format([{x['number']: x['id']} for x in reversed(tracks)]))

    def actionAddTrackBelow_trigger(self, checked=True):
        # Get selected track
        all_tracks = get_app().project.get("layers")
        selected_layer_id = self.selected_tracks[0]

        log.info("adding track below %s", selected_layer_id)

        # Get track data for selected track
        existing_track = Track.get(id=selected_layer_id)
        if not existing_track:
            # Log error and fail silently
            log.error('No track object found with id: %s', selected_layer_id)
            return
        selected_layer_num = int(existing_track.data["number"])

        # Get track below selected track (if any)
        try:
            tracks = sorted(all_tracks, key=lambda x: x['number'])
            existing_index = tracks.index(existing_track.data)
        except ValueError:
            log.warning("Could not find track %s", selected_layer_num, exc_info=1)
            return

        if existing_index > 0:
            prev_index = existing_index - 1
            prev_layer = tracks[prev_index]
            delta = abs(selected_layer_num - prev_layer.get('number'))
        else:
            delta = selected_layer_num

        # Calculate new track number (based on gap delta)
        if delta > 2:
            # New track number (pick mid point in track number gap)
            new_track_num = selected_layer_num - int(round(delta / 2.0))

            log.info("New track num %s (delta %s)", new_track_num, delta)

            # Create new track and insert
            track = Track()
            track.data = {"number": new_track_num, "y": 0, "label": "", "lock": False}
            track.save()
        else:
            # Track numbering is too tight, renumber them all and insert
            self.renumber_all_layers(insert_at=existing_index)

        tracks = sorted(get_app().project.get("layers"), key=lambda x: x['number'])

        # Temporarily for debugging
        log.info("Tracks after: {}".format([{x['number']: x['id']} for x in reversed(tracks)]))

    def actionArrowTool_trigger(self, checked=True):
        log.info("actionArrowTool_trigger")

    def actionSnappingTool_trigger(self, checked=True):
        log.info("actionSnappingTool_trigger")

        # Enable / Disable snapping mode
        self.timeline.SetSnappingMode(self.actionSnappingTool.isChecked())

    def actionRazorTool_trigger(self, checked=True):
        """Toggle razor tool on and off"""
        log.info('actionRazorTool_trigger')

        # Enable / Disable razor mode
        self.timeline.SetRazorMode(checked)

    def actionAddMarker_trigger(self, checked=True):
        log.info("actionAddMarker_trigger")

        # Get player object
        player = self.preview_thread.player

        # Calculate frames per second
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])

        # Calculate position in seconds
        position = (player.Position() - 1) / fps_float

        # Look for existing Marker
        marker = Marker()
        marker.data = {"position": position, "icon": "blue.png", "type": "orig", "frameNo": player.Position()}
        marker.save()

    ### CHANGED ###

    class LipSync_thread(QObject):
        finished = pyqtSignal()
        progress = pyqtSignal(str)

        def __init__(self, pos, url_ls, box):
            QObject.__init__(self)
            self.pos = pos
            self.url_ls = url_ls
            self.box = box

        def run(self):

            try:
                #print(self.pos,"Okay")
                log.info("In Lipsync thread")

                name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 7)))
                ls_pos = sorted(self.pos['ls'])
                #print(ls_pos,ls_pos[1:-1],"lame")
                fps = get_app().project.get("fps")
                fps_float = float(fps["num"]) / float(fps["den"])


                #----change---------------
                if len(ls_pos)==2:
                    ls_pos = [int(fps_float * x + 1) for x in ls_pos]
                else:
                    ls_pos = [int(fps_float * x + 1) for x in ls_pos[1:-1]]
                # print(ls_pos)
                from windows.export import Export
                win = Export()
                print(ls_pos, name)
                if len(ls_pos)==0:
                    fn, epn = win.accept(True, name)
                else:
                    fn, epn = win.accept(ls_pos[0], ls_pos[1], True, name)
                print(ls_pos,name, fn, epn)
                
                outdir = os.path.join(sys.path[0], 'ls_out', 'to_send')

                pth1 = os.path.join(outdir, name + '.mp4')
                #_pth1 = os.path.join(outdir, name + '_1.mp4')
                pth2 = os.path.join(outdir, name + '.wav')
                video_cmd = 'ffmpeg -i {} -c copy -an {}'.format(epn, pth1) # -an: no audio -c copy: copy video -i: input
                print(video_cmd)
                audio_cmd = 'ffmpeg -i {} {}'.format(epn, pth2)

                subprocess.call(video_cmd, shell=True)
                #cmd = 'ffmpeg -i {} {}'.format(pth1,_pth1)
                subprocess.call(audio_cmd, shell=True)
                #subprocess.call(cmd, shell=True)
        
                outfile = os.path.join(sys.path[0], 'ls_out', 'res', name + '.mp4')

                cmd = 'curl --max-time 800 -X POST -F file1=@{} -F file2=@{} {}/{} > {}'.format(pth1, pth2, self.url_ls, int(fps_float), outfile)
                print(pth1, pth2, self.url_ls, int(fps_float), outfile,"life")
                print(self.box)
                if self.box[0] != -1:
                    pth3 = os.path.join(outdir, name + '.txt')
                    with open(pth3, 'w') as f:
                        f.write("{} {} {} {}".format(self.box[0], self.box[1], self.box[2], self.box[3]))
                    f.close()
                    cmd = 'curl --max-time 800 -X POST -F file1=@{} -F file2=@{} -F file3=@{} {}/{} > {}'.format(pth1, pth2, pth3, self.url_ls, int(fps_float), outfile)
                
                subprocess.call(cmd, shell=True)
            except Exception as e:
                log.info(e)
                outfile = "None"
                print("Faced some error in lipSync thread")        
            # sleep(5);
            self.progress.emit(outfile)
            self.finished.emit()

    def actionLipSync_trigger(self, checked=True):
        log.info("actionLipSync_trigger")


        self.LSthread = QThread()
        self.LSworker = self.LipSync_thread(self.findAllMarkerPositions(), self.url_ls, self.coordinates);
        self.LSworker.moveToThread(self.LSthread)
        self.LSthread.started.connect(self.LSworker.run)
        self.LSworker.finished.connect(self.LSthread.quit)
        self.LSworker.finished.connect(self.LSworker.deleteLater)
        self.LSthread.finished.connect(self.LSthread.deleteLater)
        self.LSworker.progress.connect(self.LSfinished)

        self.LSthread.start()
        self.lipSyncMsgBox = QMessageBox()
        self.lipSyncMsgBox.setWindowTitle('Work in progess')
        self.lipSyncMsgBox.setText('Executing Lipsync...')
        # self.lipSyncMsgBox.addButton(QMessageBox.Ok)
        self.lipSyncMsgBox.show()

        self.LSthread.finished.connect(
            lambda: self.lipSyncMsgBox.hide()
        )

    def LSfinished(self, filename):
        qfilename = QUrl.fromLocalFile(filename)
        self.files_model.process_urls([qfilename])        


    def actionAddMarker_ls_trigger(self, checked=True):
        log.info("actionAddMarker_trigger_ls")

        # Get player object
        player = self.preview_thread.player

        # Calculate frames per second
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])

        # Calculate position in seconds
        position = (player.Position() - 1) / fps_float

        # Look for existing Marker
        marker = Marker()
        marker.data = {"position": position, "icon": "red.png", "type": "ls", "frameNo": player.Position()}
        marker.save()        

    def actionAddMarker_sn_trigger(self, checked=True):
        log.info("actionAddMarker_trigger_sn")

        # Get player object
        player = self.preview_thread.player

        # Calculate frames per second
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])

        # Calculate position in seconds
        position = (player.Position() - 1) / fps_float

        # Look for existing Marker
        marker = Marker()
        marker.data = {"position": position, "icon": "yellow.png", "type": "sn", "frameNo": player.Position()}
        marker.save()



    def shape_selection(self, event, x, y, flags, param):
        '''
        Function for drawing rectangle
        '''
        global ref_point

        if event == cv2.EVENT_LBUTTONDOWN: 
            ref_point = [(x, y)] 

        elif event == cv2.EVENT_LBUTTONUP: 
            ref_point.append((x, y))
            # image = param[0]
            cv2.rectangle(self.image, ref_point[0], ref_point[1], (0, 0, 255), 2)
            cv2.imshow("image", self.image)

    def actionSyncNet_trigger(self, checked=True):
        log.info("actionSyncNet_trigger")
        # cv2.imshow(cv2.imread('/home/anchit/Pictures/aa.png'))
        # a = openshot.Timeline.GetFrame(self.timeline_sync.timeline, self.preview_thread.current_frame)
        img = self.videoPreview.current_image
        size = img.size()
        image_width, image_height, image_depth = size.width(), size.height(), img.depth()
        s = img.bits().asstring(image_width * image_height * image_depth // 8)  # format 0xffRRGGBB

        arr = np.fromstring(s, dtype=np.uint8).reshape((image_height, image_width, image_depth // 8))
        # s = image.bits().asstring(self.videoPreview.width() * self.videoPreview.height() * 4)
        # arr = np.fromstring(s, dtype=np.uint8).reshape((self.videoPreview.height(), self.videoPreview.width(), 4))
        # print(arr)
        self.image = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.shape_selection)

        while True: 
            cv2.imshow("image", self.image) 
            key = cv2.waitKey(1) & 0xFF

            # press key r to redraw rectangle
            if key == ord("r"): 
                self.image = clone.copy()

            # press key c to capture
            elif key == ord("c"): 
                break
        global ref_point

        # if len(ref_point) == 2: 
        #     crop_img1 = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]: 
        #                                                         ref_point[1][0]] 
        #     cv2.imshow("crop_img", crop_img1)
        #     cv2.waitKey(0)

        app = get_app()
        height = app.project.get('height')
        width = app.project.get('width')
        print(image_height, height, image_width, width)
        height_ratio = height/image_height
        width_ratio = width/image_width
        ref_point = np.array(ref_point)
        print(ref_point, height_ratio, width_ratio)
        x1 = int(height_ratio * ref_point[0][0])
        x2 = int(height_ratio * ref_point[1][0])
        y1 = int(width_ratio * ref_point[0][1])
        y2 = int(width_ratio * ref_point[1][1])
        # self.coordinates = [x1, y1, x2, y2]
        self.coordinates = [y1, y2, x1, x2]

    ### CHANGED ###

    def findAllMarkerPositions(self):
        """Build and return a list of all seekable locations for the currently-selected timeline elements"""

        def getTimelineObjectPositions(obj):
            """ Add boundaries & all keyframes of a timeline object (clip, transition...) to all_marker_positions """
            positions = []

            fps = get_app().project.get("fps")
            fps_float = float(fps["num"]) / float(fps["den"])

            clip_start_time = obj.data["position"]
            clip_orig_time = clip_start_time - obj.data["start"]
            clip_stop_time = clip_orig_time + obj.data["end"]

            # add clip boundaries
            positions.append(clip_start_time)
            positions.append(clip_stop_time)

            # add all keyframes
            for property in obj.data:
                try:
                    for point in obj.data[property]["Points"]:
                        keyframe_time = (point["co"]["X"]-1)/fps_float - obj.data["start"] + obj.data["position"]
                        if clip_start_time < keyframe_time < clip_stop_time:
                            positions.append(keyframe_time)
                except (TypeError, KeyError):
                    pass

            # Add all Effect keyframes
            if "effects" in obj.data:
                for effect_data in obj.data["effects"]:
                    for prop in effect_data:
                        try:
                            for point in effect_data[prop]["Points"]:
                                keyframe_time = (point["co"]["X"]-1)/fps_float + clip_orig_time
                                if clip_start_time < keyframe_time < clip_stop_time:
                                    positions.append(keyframe_time)
                        except (TypeError, KeyError):
                            pass

            return positions

        # We can always jump to the beginning of the timeline
        marker_keys = ['orig', 'ls', 'sn']
        # all_marker_positions = [0]
        all_marker_positions = {}
        for mk in marker_keys:
            all_marker_positions[mk] = [0]

        # If nothing is selected, also add the end of the last clip
        # if not self.selected_clips + self.selected_transitions:
        for mk in marker_keys:
            all_marker_positions[mk].append(
                get_app().window.timeline_sync.timeline.GetMaxTime())

        # Get list of marker and important positions (like selected clip bounds)
        for marker in Marker.filter():
            all_marker_positions[marker.data["type"]].append(marker.data["position"])

        # # Loop through selected clips (and add key positions)
        # for clip_id in self.selected_clips:
        #     # Get selected object
        #     selected_clip = Clip.get(id=clip_id)
        #     if selected_clip:
        #         all_marker_positions[selected_clip.data["type"]].extend(getTimelineObjectPositions(selected_clip))

        # # Loop through selected transitions (and add key positions)
        # for tran_id in self.selected_transitions:
        #     # Get selected object
        #     selected_tran = Transition.get(id=tran_id)
        #     if selected_tran:
        #         all_marker_positions[selected_tran.data["type"]].extend(getTimelineObjectPositions(selected_tran))

        # remove duplicates
        # all_marker_positions = list(set(all_marker_positions))
        for mk in marker_keys:
            all_marker_positions[mk] = list(set(all_marker_positions[mk]))

        return all_marker_positions

    def actionPreviousMarker_trigger(self, checked=True):
        log.info("actionPreviousMarker_trigger")

        # Calculate current position (in seconds)
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])
        current_position = (self.preview_thread.current_frame - 1) / fps_float
        # all_marker_positions = self.findAllMarkerPositions()
        ret = self.findAllMarkerPositions()
        all_marker_positions = []
        for mk in ret:
            all_marker_positions += ret[mk]
        all_marker_positions = sorted(all_marker_positions)
        # Loop through all markers, and find the closest one to the left
        closest_position = None
        for marker_position in sorted(all_marker_positions):
            # Is marker smaller than position?
            if marker_position < current_position and (abs(marker_position - current_position) > 0.1):
                # Is marker larger than previous marker
                if closest_position and marker_position > closest_position:
                    # Set a new closest marker
                    closest_position = marker_position
                elif not closest_position:
                    # First one found
                    closest_position = marker_position

        # Seek to marker position (if any)
        if closest_position is not None:
            # Seek
            frame_to_seek = round(closest_position * fps_float) + 1
            self.SeekSignal.emit(frame_to_seek)

            # Update the preview and reselct current frame in properties
            get_app().window.refreshFrameSignal.emit()
            get_app().window.propertyTableView.select_frame(frame_to_seek)

    def actionNextMarker_trigger(self, checked=True):
        log.info("actionNextMarker_trigger")

        # Calculate current position (in seconds)
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])
        current_position = (self.preview_thread.current_frame - 1) / fps_float
        # all_marker_positions = self.findAllMarkerPositions()
        ret = self.findAllMarkerPositions()
        all_marker_positions = []
        for mk in ret:
            all_marker_positions += ret[mk]
        all_marker_positions = sorted(all_marker_positions)
        # Loop through all markers, and find the closest one to the right
        closest_position = None
        for marker_position in sorted(all_marker_positions):
            # Is marker smaller than position?
            if marker_position > current_position and (abs(marker_position - current_position) > 0.1):
                # Is marker larger than previous marker
                if closest_position and marker_position < closest_position:
                    # Set a new closest marker
                    closest_position = marker_position
                elif not closest_position:
                    # First one found
                    closest_position = marker_position

        # Seek to marker position (if any)
        if closest_position is not None:
            # Seek
            frame_to_seek = round(closest_position * fps_float) + 1
            self.SeekSignal.emit(frame_to_seek)

            # Update the preview and reselct current frame in properties
            get_app().window.refreshFrameSignal.emit()
            get_app().window.propertyTableView.select_frame(frame_to_seek)

    def actionCenterOnPlayhead_trigger(self, checked=True):
        """ Center the timeline on the current playhead position """
        self.timeline.centerOnPlayhead()

    def getShortcutByName(self, setting_name):
        """ Get a key sequence back from the setting name """
        s = get_app().get_settings()
        shortcut = QKeySequence(s.get(setting_name))
        return shortcut

    def getAllKeyboardShortcuts(self):
        """ Get a key sequence back from the setting name """
        keyboard_shortcuts = []
        all_settings = get_app().get_settings()._data
        for setting in all_settings:
            if setting.get('category') == 'Keyboard':
                keyboard_shortcuts.append(setting)
        return keyboard_shortcuts

    def keyPressEvent(self, event):
        """ Process key press events and match with known shortcuts"""
        # Detect the current KeySequence pressed (including modifier keys)
        key_value = event.key()
        modifiers = int(event.modifiers())

        # Abort handling if the key sequence is invalid
        if (key_value <= 0 or key_value in
           [Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Control, Qt.Key_Meta]):
            return

        # A valid keysequence was detected
        event.accept()
        key = QKeySequence(modifiers + key_value)

        # Get the video player object
        player = self.preview_thread.player

        # Get framerate
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])
        playhead_position = float(self.preview_thread.current_frame - 1) / fps_float

        # Basic shortcuts i.e just a letter
        if key.matches(self.getShortcutByName("seekPreviousFrame")) == QKeySequence.ExactMatch:
            # Pause video
            self.actionPlay_trigger(force="pause")
            # Set speed to 0
            if player.Speed() != 0:
                self.SpeedSignal.emit(0)
            # Seek to previous frame
            self.SeekSignal.emit(player.Position() - 1)

            # Notify properties dialog
            self.propertyTableView.select_frame(player.Position())

        elif key.matches(self.getShortcutByName("seekNextFrame")) == QKeySequence.ExactMatch:
            # Pause video
            self.actionPlay_trigger(force="pause")
            # Set speed to 0
            if player.Speed() != 0:
                self.SpeedSignal.emit(0)
            # Seek to next frame
            self.SeekSignal.emit(player.Position() + 1)

            # Notify properties dialog
            self.propertyTableView.select_frame(player.Position())

        elif key.matches(self.getShortcutByName("rewindVideo")) == QKeySequence.ExactMatch:
            # Toggle rewind and start playback
            self.actionRewind.trigger()
            ui_util.setup_icon(self, self.actionPlay, "actionPlay", "media-playback-pause")
            self.actionPlay.setChecked(True)

        elif key.matches(self.getShortcutByName("fastforwardVideo")) == QKeySequence.ExactMatch:
            # Toggle fastforward button and start playback
            self.actionFastForward.trigger()
            ui_util.setup_icon(self, self.actionPlay, "actionPlay", "media-playback-pause")
            self.actionPlay.setChecked(True)

        elif any([
                key.matches(self.getShortcutByName("playToggle")) == QKeySequence.ExactMatch,
                key.matches(self.getShortcutByName("playToggle1")) == QKeySequence.ExactMatch,
                key.matches(self.getShortcutByName("playToggle2")) == QKeySequence.ExactMatch,
                key.matches(self.getShortcutByName("playToggle3")) == QKeySequence.ExactMatch,
                ]):
            # Toggle playbutton and show properties
            self.actionPlay.trigger()
            self.propertyTableView.select_frame(player.Position())

        elif any([
                key.matches(self.getShortcutByName("deleteItem")) == QKeySequence.ExactMatch,
                key.matches(self.getShortcutByName("deleteItem1")) == QKeySequence.ExactMatch,
                ]):
            # Delete selected clip / transition
            self.actionRemoveClip.trigger()
            self.actionRemoveTransition.trigger()

        # Menu shortcuts
        elif key.matches(self.getShortcutByName("actionNew")) == QKeySequence.ExactMatch:
            self.actionNew.trigger()
        elif key.matches(self.getShortcutByName("actionOpen")) == QKeySequence.ExactMatch:
            self.actionOpen.trigger()
        elif key.matches(self.getShortcutByName("actionSave")) == QKeySequence.ExactMatch:
            self.actionSave.trigger()
        elif key.matches(self.getShortcutByName("actionUndo")) == QKeySequence.ExactMatch:
            self.actionUndo.trigger()
        elif key.matches(self.getShortcutByName("actionSaveAs")) == QKeySequence.ExactMatch:
            self.actionSaveAs.trigger()
        elif key.matches(self.getShortcutByName("actionImportFiles")) == QKeySequence.ExactMatch:
            self.actionImportFiles.trigger()
        elif key.matches(self.getShortcutByName("actionRedo")) == QKeySequence.ExactMatch:
            self.actionRedo.trigger()
        elif key.matches(self.getShortcutByName("actionExportVideo")) == QKeySequence.ExactMatch:
            self.actionExportVideo.trigger()
        elif key.matches(self.getShortcutByName("actionQuit")) == QKeySequence.ExactMatch:
            self.actionQuit.trigger()
        elif key.matches(self.getShortcutByName("actionPreferences")) == QKeySequence.ExactMatch:
            self.actionPreferences.trigger()
        elif key.matches(self.getShortcutByName("actionAddTrack")) == QKeySequence.ExactMatch:
            self.actionAddTrack.trigger()
        elif key.matches(self.getShortcutByName("actionAddMarker")) == QKeySequence.ExactMatch:
            self.actionAddMarker.trigger()
        elif key.matches(self.getShortcutByName("actionPreviousMarker")) == QKeySequence.ExactMatch:
            self.actionPreviousMarker.trigger()
        elif key.matches(self.getShortcutByName("actionNextMarker")) == QKeySequence.ExactMatch:
            self.actionNextMarker.trigger()
        elif key.matches(self.getShortcutByName("actionCenterOnPlayhead")) == QKeySequence.ExactMatch:
            self.actionCenterOnPlayhead.trigger()
        elif key.matches(self.getShortcutByName("actionTimelineZoomIn")) == QKeySequence.ExactMatch:
            self.actionTimelineZoomIn.trigger()
        elif key.matches(self.getShortcutByName("actionTimelineZoomOut")) == QKeySequence.ExactMatch:
            self.actionTimelineZoomOut.trigger()
        elif key.matches(self.getShortcutByName("actionTitle")) == QKeySequence.ExactMatch:
            self.actionTitle.trigger()
        elif key.matches(self.getShortcutByName("actionAnimatedTitle")) == QKeySequence.ExactMatch:
            self.actionAnimatedTitle.trigger()
        elif key.matches(self.getShortcutByName("actionDuplicateTitle")) == QKeySequence.ExactMatch:
            self.actionDuplicateTitle.trigger()
        elif key.matches(self.getShortcutByName("actionEditTitle")) == QKeySequence.ExactMatch:
            self.actionEditTitle.trigger()
        elif key.matches(self.getShortcutByName("actionFullscreen")) == QKeySequence.ExactMatch:
            self.actionFullscreen.trigger()
        elif key.matches(self.getShortcutByName("actionAbout")) == QKeySequence.ExactMatch:
            self.actionAbout.trigger()
        elif key.matches(self.getShortcutByName("actionThumbnailView")) == QKeySequence.ExactMatch:
            self.actionThumbnailView.trigger()
        elif key.matches(self.getShortcutByName("actionDetailsView")) == QKeySequence.ExactMatch:
            self.actionDetailsView.trigger()
        elif key.matches(self.getShortcutByName("actionProfile")) == QKeySequence.ExactMatch:
            self.actionProfile.trigger()
        elif key.matches(self.getShortcutByName("actionAdd_to_Timeline")) == QKeySequence.ExactMatch:
            self.actionAdd_to_Timeline.trigger()
        elif key.matches(self.getShortcutByName("actionSplitClip")) == QKeySequence.ExactMatch:
            self.actionSplitClip.trigger()
        elif key.matches(self.getShortcutByName("actionSnappingTool")) == QKeySequence.ExactMatch:
            self.actionSnappingTool.trigger()
        elif key.matches(self.getShortcutByName("actionJumpStart")) == QKeySequence.ExactMatch:
            self.actionJumpStart.trigger()
        elif key.matches(self.getShortcutByName("actionJumpEnd")) == QKeySequence.ExactMatch:
            self.actionJumpEnd.trigger()
        elif key.matches(self.getShortcutByName("actionSaveFrame")) == QKeySequence.ExactMatch:
            self.actionSaveFrame.trigger()
        elif key.matches(self.getShortcutByName("actionProperties")) == QKeySequence.ExactMatch:
            self.actionProperties.trigger()
        elif key.matches(self.getShortcutByName("actionTransform")) == QKeySequence.ExactMatch:
            if self.selected_clips:
                self.TransformSignal.emit(self.selected_clips[0])
        elif key.matches(self.getShortcutByName("actionInsertKeyframe")) == QKeySequence.ExactMatch:
            log.debug("actionInsertKeyframe")
            if self.selected_clips or self.selected_transitions:
                self.InsertKeyframe.emit(event)

        # Timeline keyboard shortcuts
        elif key.matches(self.getShortcutByName("sliceAllKeepBothSides")) == QKeySequence.ExactMatch:
            intersecting_clips = Clip.filter(intersect=playhead_position)
            intersecting_trans = Transition.filter(intersect=playhead_position)
            if intersecting_clips or intersecting_trans:
                # Get list of clip ids
                clip_ids = [c.id for c in intersecting_clips]
                trans_ids = [t.id for t in intersecting_trans]
                self.timeline.Slice_Triggered(0, clip_ids, trans_ids, playhead_position)
        elif key.matches(self.getShortcutByName("sliceAllKeepLeftSide")) == QKeySequence.ExactMatch:
            intersecting_clips = Clip.filter(intersect=playhead_position)
            intersecting_trans = Transition.filter(intersect=playhead_position)
            if intersecting_clips or intersecting_trans:
                # Get list of clip ids
                clip_ids = [c.id for c in intersecting_clips]
                trans_ids = [t.id for t in intersecting_trans]
                self.timeline.Slice_Triggered(1, clip_ids, trans_ids, playhead_position)
        elif key.matches(self.getShortcutByName("sliceAllKeepRightSide")) == QKeySequence.ExactMatch:
            intersecting_clips = Clip.filter(intersect=playhead_position)
            intersecting_trans = Transition.filter(intersect=playhead_position)
            if intersecting_clips or intersecting_trans:
                # Get list of clip ids
                clip_ids = [c.id for c in intersecting_clips]
                trans_ids = [t.id for t in intersecting_trans]
                self.timeline.Slice_Triggered(2, clip_ids, trans_ids, playhead_position)
        elif key.matches(self.getShortcutByName("sliceSelectedKeepBothSides")) == QKeySequence.ExactMatch:
            intersecting_clips = Clip.filter(intersect=playhead_position)
            intersecting_trans = Transition.filter(intersect=playhead_position)
            if intersecting_clips or intersecting_trans:
                # Get list of clip ids
                clip_ids = [c.id for c in intersecting_clips if c.id in self.selected_clips]
                trans_ids = [t.id for t in intersecting_trans if t.id in self.selected_transitions]
                self.timeline.Slice_Triggered(0, clip_ids, trans_ids, playhead_position)
        elif key.matches(self.getShortcutByName("sliceSelectedKeepLeftSide")) == QKeySequence.ExactMatch:
            intersecting_clips = Clip.filter(intersect=playhead_position)
            intersecting_trans = Transition.filter(intersect=playhead_position)
            if intersecting_clips or intersecting_trans:
                # Get list of clip ids
                clip_ids = [c.id for c in intersecting_clips if c.id in self.selected_clips]
                trans_ids = [t.id for t in intersecting_trans if t.id in self.selected_transitions]
                self.timeline.Slice_Triggered(1, clip_ids, trans_ids, playhead_position)
        elif key.matches(self.getShortcutByName("sliceSelectedKeepRightSide")) == QKeySequence.ExactMatch:
            intersecting_clips = Clip.filter(intersect=playhead_position)
            intersecting_trans = Transition.filter(intersect=playhead_position)
            if intersecting_clips or intersecting_trans:
                # Get list of ids that are also selected
                clip_ids = [c.id for c in intersecting_clips if c.id in self.selected_clips]
                trans_ids = [t.id for t in intersecting_trans if t.id in self.selected_transitions]
                self.timeline.Slice_Triggered(2, clip_ids, trans_ids, playhead_position)

        elif key.matches(self.getShortcutByName("copyAll")) == QKeySequence.ExactMatch:
            self.timeline.Copy_Triggered(-1, self.selected_clips, self.selected_transitions)
        elif key.matches(self.getShortcutByName("pasteAll")) == QKeySequence.ExactMatch:
            self.timeline.Paste_Triggered(9, float(playhead_position), -1, [], [])
        elif key.matches(self.getShortcutByName("nudgeLeft")) == QKeySequence.ExactMatch:
            self.timeline.Nudge_Triggered(-1, self.selected_clips, self.selected_transitions)
        elif key.matches(self.getShortcutByName("nudgeRight")) == QKeySequence.ExactMatch:
            self.timeline.Nudge_Triggered(1, self.selected_clips, self.selected_transitions)

        # Select All / None
        elif key.matches(self.getShortcutByName("selectAll")) == QKeySequence.ExactMatch:
            self.timeline.SelectAll()

        elif key.matches(self.getShortcutByName("selectNone")) == QKeySequence.ExactMatch:
            self.timeline.ClearAllSelections()

        # If we didn't act on the event, forward it to the base class
        else:
            super().keyPressEvent(event)

    def actionProfile_trigger(self):
        # Show dialog
        from windows.profile import Profile
        log.debug("Showing preferences dialog")
        win = Profile()
        # Run the dialog event loop - blocking interaction on this window during this time
        win.exec_()
        log.debug("Preferences dialog closed")

    def actionSplitClip_trigger(self):
        log.debug("actionSplitClip_trigger")

        # Loop through selected files (set 1 selected file if more than 1)
        f = self.files_model.current_file()

        # Bail out if no file selected
        if not f:
            log.warn("Split clip action failed, couldn't find current file")
            return

        # show dialog
        from windows.cutting import Cutting
        win = Cutting(f)
        # Run the dialog event loop - blocking interaction on this window during that time
        result = win.exec_()
        if result == QDialog.Accepted:
            log.info('Cutting Finished')
        else:
            log.info('Cutting Cancelled')

    def actionRemove_from_Project_trigger(self):
        log.debug("actionRemove_from_Project_trigger")

        # Loop through selected files
        for f in self.selected_files():
            if not f:
                continue

            f_id = f.data["id"]
            # Remove file
            f.delete()

            # Find matching clips (if any)
            clips = Clip.filter(file_id=f_id)
            for c in clips:
                # Remove clip
                c.delete()

        # Refresh preview
        get_app().window.refreshFrameSignal.emit()

    def actionRemoveClip_trigger(self):
        log.debug('actionRemoveClip_trigger')

        # Loop through selected clips
        for clip_id in deepcopy(self.selected_clips):
            # Find matching file
            clips = Clip.filter(id=clip_id)
            for c in clips:
                # Clear selected clips
                self.removeSelection(clip_id, "clip")

                # Remove clip
                c.delete()

        # Refresh preview
        get_app().window.refreshFrameSignal.emit()

    def actionProperties_trigger(self):
        log.debug('actionProperties_trigger')

        # Show properties dock
        if not self.dockProperties.isVisible():
            self.dockProperties.show()

    def actionRemoveEffect_trigger(self):
        log.debug('actionRemoveEffect_trigger')

        # Loop through selected clips
        for effect_id in deepcopy(self.selected_effects):
            log.info("effect id: %s" % effect_id)

            # Find matching file
            clips = Clip.filter()
            found_effect = None
            for c in clips:
                found_effect = False
                log.info("c.data[effects]: %s" % c.data["effects"])

                for effect in c.data["effects"]:
                    if effect["id"] == effect_id:
                        found_effect = effect
                        break

                if found_effect:
                    # Remove found effect from clip data and save clip
                    c.data["effects"].remove(found_effect)

                    # Remove unneeded attributes from JSON
                    c.data.pop("reader")

                    # Save clip
                    c.save()

                    # Clear selected effects
                    self.removeSelection(effect_id, "effect")

        # Refresh preview
        self.refreshFrameSignal.emit()

    def actionRemoveTransition_trigger(self):
        log.debug('actionRemoveTransition_trigger')

        # Loop through selected clips
        for tran_id in deepcopy(self.selected_transitions):
            # Find matching file
            transitions = Transition.filter(id=tran_id)
            for t in transitions:
                # Clear selected clips
                self.removeSelection(tran_id, "transition")

                # Remove transition
                t.delete()

        # Refresh preview
        self.refreshFrameSignal.emit()

    def actionRemoveTrack_trigger(self):
        log.debug('actionRemoveTrack_trigger')

        # Get translation function
        _ = get_app()._tr

        track_id = self.selected_tracks[0]
        max_track_number = len(get_app().project.get("layers"))

        # Get details of selected track
        selected_track = Track.get(id=track_id)
        selected_track_number = int(selected_track.data["number"])

        # Don't allow user to delete final track
        if max_track_number == 1:
            # Show error and do nothing
            QMessageBox.warning(self, _("Error Removing Track"), _("You must keep at least 1 track"))
            return

        # Revove all clips on this track first
        for clip in Clip.filter(layer=selected_track_number):
            clip.delete()

        # Revove all transitions on this track first
        for trans in Transition.filter(layer=selected_track_number):
            trans.delete()

        # Remove track
        selected_track.delete()

        # Clear selected track
        self.selected_tracks = []

        # Refresh preview
        self.refreshFrameSignal.emit()

    def actionLockTrack_trigger(self):
        """Callback for locking a track"""
        log.debug('actionLockTrack_trigger')

        # Get details of track
        track_id = self.selected_tracks[0]
        selected_track = Track.get(id=track_id)

        # Lock track and save
        selected_track.data['lock'] = True
        selected_track.save()

    def actionUnlockTrack_trigger(self):
        """Callback for unlocking a track"""
        log.info('actionUnlockTrack_trigger')

        # Get details of track
        track_id = self.selected_tracks[0]
        selected_track = Track.get(id=track_id)

        # Lock track and save
        selected_track.data['lock'] = False
        selected_track.save()

    def actionRenameTrack_trigger(self):
        """Callback for renaming track"""
        log.info('actionRenameTrack_trigger')

        # Get translation function
        _ = get_app()._tr

        # Get details of track
        track_id = self.selected_tracks[0]
        selected_track = Track.get(id=track_id)

        # Find display track number
        all_tracks = get_app().project.get("layers")
        display_count = len(all_tracks)
        for track in reversed(sorted(all_tracks, key=lambda x: x['number'])):
            if track.get("id") == track_id:
                break
            display_count -= 1

        track_name = selected_track.data["label"] or _("Track %s") % display_count

        text, ok = QInputDialog.getText(self, _('Rename Track'), _('Track Name:'), text=track_name)
        if ok:
            # Update track
            selected_track.data["label"] = text
            selected_track.save()

    def actionRemoveAllMarkers_trigger(self):
        log.info('actionRemoveAllMarkers_trigger')
        # print(Marker.filter())
        for m in Marker.filter():
            m.delete()

    def actionRemoveMarker_trigger(self):
        log.info('actionRemoveMarker_trigger')

        for marker_id in self.selected_markers:
            marker = Marker.filter(id=marker_id)
            for m in marker:
                # Remove track
                m.delete()

    def actionTimelineZoomIn_trigger(self):
        self.sliderZoomWidget.zoomIn()

    def actionTimelineZoomOut_trigger(self):
        self.sliderZoomWidget.zoomOut()

    def actionFullscreen_trigger(self):
        # Toggle fullscreen state (current state mask XOR WindowFullScreen)
        self.setWindowState(self.windowState() ^ Qt.WindowFullScreen)

    def actionFile_Properties_trigger(self):
        log.info("Show file properties")

        # Get current selected file (corresponding to menu, if possible)
        f = self.files_model.current_file()
        if not f:
            log.warning("Couldn't find current file for properties window")
            return

        # show dialog
        from windows.file_properties import FileProperties
        win = FileProperties(f)
        # Run the dialog event loop - blocking interaction on this window during that time
        result = win.exec_()
        if result == QDialog.Accepted:

            # BRUTE FORCE approach: go through all clips and update file path
            clips = Clip.filter(file_id=f.data["id"])
            for c in clips:
                # update clip
                c.data["reader"]["path"] = f.data["path"]
                c.save()

            log.info('File Properties Finished')
        else:
            log.info('File Properties Cancelled')

    def actionDetailsView_trigger(self):
        log.info("Switch to Details View")

        # Get settings
        app = get_app()
        s = app.get_settings()

        # Files
        if app.context_menu_object == "files":
            s.set("file_view", "details")
            self.filesListView.hide()
            self.filesView = self.filesTreeView
            self.filesView.show()

        # Transitions
        elif app.context_menu_object == "transitions":
            s.set("transitions_view", "details")
            self.transitionsListView.hide()
            self.transitionsView = self.transitionsTreeView
            self.transitionsView.show()

        # Effects
        elif app.context_menu_object == "effects":
            s.set("effects_view", "details")
            self.effectsListView.hide()
            self.effectsView = self.effectsTreeView
            self.effectsView.show()

    def actionThumbnailView_trigger(self):
        log.info("Switch to Thumbnail View")

        # Get settings
        app = get_app()
        s = app.get_settings()

        # Files
        if app.context_menu_object == "files":
            s.set("file_view", "thumbnail")
            self.filesTreeView.hide()
            self.filesView = self.filesListView
            self.filesView.show()

        # Transitions
        elif app.context_menu_object == "transitions":
            s.set("transitions_view", "thumbnail")
            self.transitionsTreeView.hide()
            self.transitionsView = self.transitionsListView
            self.transitionsView.show()

        # Effects
        elif app.context_menu_object == "effects":
            s.set("effects_view", "thumbnail")
            self.effectsTreeView.hide()
            self.effectsView = self.effectsListView
            self.effectsView.show()

    def resize_contents(self):
        if self.filesView == self.filesTreeView:
            self.filesTreeView.resize_contents()

    def getDocks(self):
        """ Get a list of all dockable widgets """
        return self.findChildren(QDockWidget)

    def removeDocks(self):
        """ Remove all dockable widgets on main screen """
        for dock in self.getDocks():
            if self.dockWidgetArea(dock) != Qt.NoDockWidgetArea:
                self.removeDockWidget(dock)

    def addDocks(self, docks, area):
        """ Add all dockable widgets to the same dock area on the main screen """
        for dock in docks:
            self.addDockWidget(area, dock)

    def floatDocks(self, is_floating):
        """ Float or Un-Float all dockable widgets above main screen """
        for dock in self.getDocks():
            if self.dockWidgetArea(dock) != Qt.NoDockWidgetArea:
                dock.setFloating(is_floating)

    def showDocks(self, docks):
        """ Show all dockable widgets on the main screen """
        for dock in docks:
            if self.dockWidgetArea(dock) != Qt.NoDockWidgetArea:
                # Only show correctly docked widgets
                dock.show()

    def freezeDocks(self):
        """ Freeze all dockable widgets on the main screen
            (prevent them being closed, floated, or moved) """
        for dock in self.getDocks():
            if self.dockWidgetArea(dock) != Qt.NoDockWidgetArea:
                dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

    def unFreezeDocks(self):
        """ Un-freeze all dockable widgets on the main screen
            (allow them to be closed, floated, or moved, as appropriate) """
        for dock in self.getDocks():
            if self.dockWidgetArea(dock) != Qt.NoDockWidgetArea:
                if dock is self.dockTimeline:
                    dock.setFeatures(
                        QDockWidget.DockWidgetFloatable
                        | QDockWidget.DockWidgetMovable)
                else:
                    dock.setFeatures(
                        QDockWidget.DockWidgetClosable
                        | QDockWidget.DockWidgetFloatable
                        | QDockWidget.DockWidgetMovable)

    def addViewDocksMenu(self):
        """ Insert a Docks submenu into the View menu """
        _ = get_app()._tr

        # self.docks_menu = self.createPopupMenu()
        # self.docks_menu.setTitle(_("Docks"))
        # self.menuView.addMenu(self.docks_menu)
        self.docks_menu = self.menuView.addMenu(_("Docks"))

        for dock in sorted(self.getDocks(), key=lambda d: d.windowTitle()):
            if (dock.features() & QDockWidget.DockWidgetClosable
               != QDockWidget.DockWidgetClosable):
                # Skip non-closable docs
                continue
            self.docks_menu.addAction(dock.toggleViewAction())

    def actionSimple_View_trigger(self):
        """ Switch to the default / simple view  """
        self.removeDocks()

        # Add Docks
        self.addDocks([
            self.dockFiles,
            self.dockTransitions,
            self.dockEffects,
            self.dockEmojis,
            self.dockVideo,
            ], Qt.TopDockWidgetArea)

        self.floatDocks(False)
        self.tabifyDockWidget(self.dockFiles, self.dockTransitions)
        self.tabifyDockWidget(self.dockTransitions, self.dockEffects)
        self.tabifyDockWidget(self.dockEffects, self.dockEmojis)
        self.showDocks([
            self.dockFiles,
            self.dockTransitions,
            self.dockEffects,
            self.dockEmojis,
            self.dockVideo,
            ])

        # Set initial size of docks
        simple_state = "".join([
            "AAAA/wAAAAD9AAAAAwAAAAAAAAEnAAAC3/wCAAAAA/wAAAJeAAAApwAAAAAA////+gAAAAACAAAAAfsAAAA"
            "YAGQAbwBjAGsASwBlAHkAZgByAGEAbQBlAAAAAAD/////AAAAAAAAAAD7AAAAHABkAG8AYwBrAFAAcgBvAH"
            "AAZQByAHQAaQBlAHMAAAAAJwAAAt8AAAChAP////sAAAAYAGQAbwBjAGsAVAB1AHQAbwByAGkAYQBsAgAAA"
            "AAAAAAAAAAAyAAAAGQAAAABAAABHAAAAUD8AgAAAAH7AAAAGABkAG8AYwBrAEsAZQB5AGYAcgBhAG0AZQEA"
            "AAFYAAAAFQAAAAAAAAAAAAAAAgAABEYAAALY/AEAAAAC/AAAAAAAAANnAAAA+gD////8AgAAAAL8AAAAJwA"
            "AAcAAAACvAP////wBAAAAAvwAAAAAAAABFQAAAHsA////+gAAAAACAAAAA/sAAAASAGQAbwBjAGsARgBpAG"
            "wAZQBzAQAAAAD/////AAAAkgD////7AAAAHgBkAG8AYwBrAFQAcgBhAG4AcwBpAHQAaQBvAG4AcwEAAAAA/"
            "////wAAAJIA////+wAAABYAZABvAGMAawBFAGYAZgBlAGMAdABzAQAAAAD/////AAAAkgD////7AAAAEgBk"
            "AG8AYwBrAFYAaQBkAGUAbwEAAAEbAAACTAAAAEcA////+wAAABgAZABvAGMAawBUAGkAbQBlAGwAaQBuAGU"
            "BAAAB7QAAARIAAACWAP////wAAANtAAAA2QAAAIIA////+gAAAAECAAAAAvsAAAAiAGQAbwBjAGsAQwBhAH"
            "AAdABpAG8AbgBFAGQAaQB0AG8AcgAAAAAA/////wAAAJgA////+wAAABQAZABvAGMAawBFAG0AbwBqAGkAc"
            "wEAAADFAAACOgAAAJIA////AAAERgAAAAEAAAABAAAAAgAAAAEAAAAC/AAAAAEAAAACAAAAAQAAAA4AdABv"
            "AG8AbABCAGEAcgEAAAAA/////wAAAAAAAAAA"
        ])
        self.restoreState(qt_types.str_to_bytes(simple_state))
        QCoreApplication.processEvents()

    def actionAdvanced_View_trigger(self):
        """ Switch to an alternative view """
        self.removeDocks()

        # Add Docks
        self.addDocks([self.dockFiles, self.dockVideo], Qt.TopDockWidgetArea)
        self.addDocks([
            self.dockEffects,
            self.dockTransitions,
            self.dockEmojis,
            ], Qt.RightDockWidgetArea)
        self.addDocks([self.dockProperties], Qt.LeftDockWidgetArea)

        self.floatDocks(False)
        self.tabifyDockWidget(self.dockEmojis, self.dockEffects)
        self.showDocks([
            self.dockFiles,
            self.dockTransitions,
            self.dockVideo,
            self.dockEffects,
            self.dockEmojis,
            self.dockProperties,
            ])

        # Set initial size of docks
        advanced_state = "".join([
            "AAAA/wAAAAD9AAAAAwAAAAAAAADxAAAC3/wCAAAAAvsAAAAcAGQAbwBjAGsAUAByAG8AcABlAHIAdABpAGUAcw"
            "EAAAAnAAAC3wAAAKEA/////AAAAl4AAACnAAAAAAD////6AAAAAAIAAAAB+wAAABgAZABvAGMAawBLAGUAeQBm"
            "AHIAYQBtAGUAAAAAAP////8AAAAAAAAAAAAAAAEAAACZAAAC3/wCAAAAAvsAAAAYAGQAbwBjAGsASwBlAHkAZg"
            "ByAGEAbQBlAQAAAVgAAAAVAAAAAAAAAAD8AAAAJwAAAt8AAAC1AQAAHPoAAAAAAQAAAAL7AAAAFgBkAG8AYwBr"
            "AEUAZgBmAGUAYwB0AHMBAAADrQAAAJkAAABYAP////sAAAAiAGQAbwBjAGsAQwBhAHAAdABpAG8AbgBFAGQAaQ"
            "B0AG8AcgEAAAAA/////wAAAFgA////AAAAAgAAArAAAALY/AEAAAAB/AAAAPcAAAKwAAAA+gD////8AgAAAAL8"
            "AAAAJwAAAcgAAAFHAP////wBAAAAAvwAAAD3AAAArgAAAIIA/////AIAAAAC+wAAABIAZABvAGMAawBGAGkAbA"
            "BlAHMBAAAAJwAAAOQAAACSAP////wAAAERAAAA3gAAAK8BAAAc+gAAAAABAAAAAvsAAAAeAGQAbwBjAGsAVABy"
            "AGEAbgBzAGkAdABpAG8AbgBzAQAAAAD/////AAAAbAD////7AAAAFABkAG8AYwBrAEUAbQBvAGoAaQBzAQAAAP"
            "cAAAEdAAAAggD////7AAAAEgBkAG8AYwBrAFYAaQBkAGUAbwEAAAGrAAAB/AAAAEcA////+wAAABgAZABvAGMA"
            "awBUAGkAbQBlAGwAaQBuAGUBAAAB9QAAAQoAAACWAP///wAAArAAAAABAAAAAQAAAAIAAAABAAAAAvwAAAABAA"
            "AAAgAAAAEAAAAOAHQAbwBvAGwAQgBhAHIBAAAAAP////8AAAAAAAAAAA=="
            ])
        self.restoreState(qt_types.str_to_bytes(advanced_state))
        QCoreApplication.processEvents()

    def actionFreeze_View_trigger(self):
        """ Freeze all dockable widgets on the main screen """
        self.freezeDocks()
        self.actionFreeze_View.setVisible(False)
        self.actionUn_Freeze_View.setVisible(True)
        self.docks_frozen = True

    def actionUn_Freeze_View_trigger(self):
        """ Un-Freeze all dockable widgets on the main screen """
        self.unFreezeDocks()
        self.actionFreeze_View.setVisible(True)
        self.actionUn_Freeze_View.setVisible(False)
        self.docks_frozen = False

    def actionShow_All_trigger(self):
        """ Show all dockable widgets """
        self.showDocks(self.getDocks())

    def actionTutorial_trigger(self):
        """ Show tutorial again """
        s = get_app().get_settings()

        # Clear tutorial settings
        s.set("tutorial_enabled", True)
        s.set("tutorial_ids", "")

        # Show first tutorial dialog again
        if self.tutorial_manager:
            self.tutorial_manager.exit_manager()
            self.tutorial_manager = TutorialManager(self)

    def actionInsertTimestamp_trigger(self, event):
        """Insert the current timestamp into the caption editor
        In the format: 00:00:23,000 --> 00:00:24,500. first click to set the initial timestamp,
        move the playehad, second click to set the end timestamp.
        """
        # Get translation function
        app = get_app()
        _ = app._tr

        if self.captionTextEdit.isReadOnly():
            return

        # Calculate fps / current seconds
        fps = get_app().project.get("fps")
        fps_float = float(fps["num"]) / float(fps["den"])
        current_position = (self.preview_thread.current_frame - 1) / fps_float

        # Get cursor / current line of text (where cursor is located)
        cursor = self.captionTextEdit.textCursor()
        self.captionTextEdit.moveCursor(QTextCursor.StartOfLine)
        line_text = cursor.block().text()
        self.captionTextEdit.moveCursor(QTextCursor.EndOfLine)

        # Insert text at cursor position
        current_timestamp = secondsToTimecode(current_position, fps["num"], fps["den"], use_milliseconds=True)
        if "-->" in line_text:
            self.captionTextEdit.insertPlainText("%s\n%s" % (current_timestamp, _("Enter caption text...")))
        else:
            self.captionTextEdit.insertPlainText("%s --> " % (current_timestamp))

    def captionTextEdit_TextChanged(self):
        """Caption text was edited, start the save timer (to prevent spamming saves)"""
        self.caption_save_timer.start()

    def caption_editor_save(self):
        """Emit the CaptionTextUpdated signal (and if that property is active/selected, it will be saved)"""
        self.CaptionTextUpdated.emit(self.captionTextEdit.toPlainText(), self.caption_model_row)

    def caption_editor_load(self, new_caption_text, caption_model_row):
        """Load the caption editor with text, or disable it if empty string detected"""
        self.caption_model_row = caption_model_row
        self.captionTextEdit.setPlainText(new_caption_text.strip())
        if not caption_model_row:
            self.captionTextEdit.setReadOnly(True)
        else:
            self.captionTextEdit.setReadOnly(False)

            # Show this dock
            self.dockCaptionEditor.show()
            self.dockCaptionEditor.raise_()

    def SetWindowTitle(self, profile=None):
        """ Set the window title based on a variety of factors """

        # Get translation function
        app = get_app()
        _ = app._tr

        if not profile:
            profile = app.project.get("profile")

        # Determine if the project needs saving (has any unsaved changes)
        save_indicator = ""
        if app.project.needs_save():
            save_indicator = "*"
            self.actionSave.setEnabled(True)
        else:
            self.actionSave.setEnabled(False)

        # Is this a saved project?
        if not app.project.current_filepath:
            # Not saved yet
            self.setWindowTitle(
                "%s %s [%s] - %s" % (
                    save_indicator,
                    _("Untitled Project"),
                    profile,
                    "OpenShot Video Editor",
                    ))
        else:
            # Yes, project is saved
            # Get just the filename
            filename = os.path.basename(app.project.current_filepath)
            filename = os.path.splitext(filename)[0]
            self.setWindowTitle(
                "%s %s [%s] - %s" % (
                    save_indicator,
                    filename,
                    profile,
                    "OpenShot Video Editor",
                    ))

    # Update undo and redo buttons enabled/disabled to available changes
    def updateStatusChanged(self, undo_status, redo_status):
        log.info('updateStatusChanged')
        self.actionUndo.setEnabled(undo_status)
        self.actionRedo.setEnabled(redo_status)
        self.SetWindowTitle()

    def addSelection(self, item_id, item_type, clear_existing=False):
        """ Add to (or clear) the selected items list for a given type. """
        if not item_id:
            log.debug('addSelection: item_type: {}, clear_existing: {}'.format(
                item_type, clear_existing))
        else:
            log.info('addSelection: item_id: {}, item_type: {}, clear_existing: {}'.format(
                item_id, item_type, clear_existing))

        s = get_app().get_settings()

        # Clear existing selection (if needed)
        if clear_existing:
            if item_type == "clip":
                self.selected_clips.clear()
                self.TransformSignal.emit("")
            elif item_type == "transition":
                self.selected_transitions.clear()
            elif item_type == "effect":
                self.selected_effects.clear()

            # Clear caption editor (if nothing is selected)
            get_app().window.CaptionTextLoaded.emit("", None)

        if item_id:
            # If item_id is not blank, store it
            if item_type == "clip" and item_id not in self.selected_clips:
                self.selected_clips.append(item_id)
                if s.get("auto-transform"):
                    self.TransformSignal.emit(self.selected_clips[-1])
            elif item_type == "transition" and item_id not in self.selected_transitions:
                self.selected_transitions.append(item_id)
            elif item_type == "effect" and item_id not in self.selected_effects:
                self.selected_effects.append(item_id)

            # Change selected item in properties view
            self.show_property_id = item_id
            self.show_property_type = item_type
            self.show_property_timer.start()

        # Notify UI that selection has been potentially changed
        self.selection_timer.start()

    # Remove from the selected items
    def removeSelection(self, item_id, item_type):
        # Remove existing selection (if any)
        if item_id:
            if item_type == "clip" and item_id in self.selected_clips:
                self.selected_clips.remove(item_id)
            elif item_type == "transition" and item_id in self.selected_transitions:
                self.selected_transitions.remove(item_id)
            elif item_type == "effect" and item_id in self.selected_effects:
                self.selected_effects.remove(item_id)

        if not self.selected_clips:
            # Clear transform (if no other clips are selected)
            self.TransformSignal.emit("")

            # Clear caption editor (if nothing is selected)
            get_app().window.CaptionTextLoaded.emit("", None)

        # Move selection to next selected clip (if any)
        self.show_property_id = ""
        self.show_property_type = ""
        if item_type == "clip" and self.selected_clips:
            self.show_property_id = self.selected_clips[0]
            self.show_property_type = item_type
        elif item_type == "transition" and self.selected_transitions:
            self.show_property_id = self.selected_transitions[0]
            self.show_property_type = item_type
        elif item_type == "effect" and self.selected_effects:
            self.show_property_id = self.selected_effects[0]
            self.show_property_type = item_type

        # Change selected item
        self.show_property_timer.start()
        self.selection_timer.start()

    def emit_selection_signal(self):
        """Emit a signal for selection changed. Callback for selection timer."""
        # Notify UI that selection has been potentially changed
        self.SelectionChanged.emit()

    def selected_files(self):
        """ Return a list of File objects for the Project Files dock's selection """
        return self.files_model.selected_files()

    def selected_file_ids(self):
        """ Return a list of File IDs for the Project Files dock's selection """
        return self.files_model.selected_file_ids()

    def current_file(self):
        """ Return the Project Files dock's currently-active item as a File object """
        return self.files_model.current_file()

    def current_file_id(self):
        """ Return the ID of the Project Files dock's currently-active item """
        return self.files_model.current_file_id()

    # Update window settings in setting store
    def save_settings(self):
        s = get_app().get_settings()

        # Save window state and geometry (saves toolbar and dock locations)
        s.set('window_state_v2', qt_types.bytes_to_str(self.saveState()))
        s.set('window_geometry_v2', qt_types.bytes_to_str(self.saveGeometry()))
        s.set('docks_frozen', self.docks_frozen)

    # Get window settings from setting store
    def load_settings(self):
        s = get_app().get_settings()

        # Window state and geometry (also toolbar, dock locations and frozen UI state)
        if s.get('window_state_v2'):
            self.restoreState(qt_types.str_to_bytes(s.get('window_state_v2')))
        if s.get('window_geometry_v2'):
            self.restoreGeometry(qt_types.str_to_bytes(s.get('window_geometry_v2')))
        if s.get('docks_frozen'):
            # Freeze all dockable widgets on the main screen
            self.freezeDocks()
            self.actionFreeze_View.setVisible(False)
            self.actionUn_Freeze_View.setVisible(True)
            self.docks_frozen = True

        # Load Recent Projects
        self.load_recent_menu()

        # The method restoreState restores the visibility of the toolBar,
        # but does not set the correct flag in the actionView_Toolbar.
        self.actionView_Toolbar.setChecked(self.toolBar.isVisibleTo(self))

    def load_recent_menu(self):
        """ Clear and load the list of recent menu items """
        s = get_app().get_settings()
        _ = get_app()._tr  # Get translation function

        # Get list of recent projects
        recent_projects = s.get("recent_projects")

        # Add Recent Projects menu (after Open File)
        import functools
        if not self.recent_menu:
            # Create a new recent menu
            self.recent_menu = self.menuFile.addMenu(
                QIcon.fromTheme("document-open-recent"),
                _("Recent Projects"))
            self.menuFile.insertMenu(self.actionRecent_Placeholder, self.recent_menu)
        else:
            # Clear the existing children
            self.recent_menu.clear()

        # Add recent projects to menu
        # Show just a placeholder menu, if we have no recent projects list
        if not recent_projects:
            self.recent_menu.addAction(_("No Recent Projects")).setDisabled(True)
            return

        for file_path in reversed(recent_projects):
            # Add each recent project
            new_action = self.recent_menu.addAction(file_path)
            new_action.triggered.connect(functools.partial(self.recent_project_clicked, file_path))

        # Add 'Clear Recent Projects' menu to bottom of list
        self.recent_menu.addSeparator()
        self.recent_menu.addAction(self.actionClearRecents)
        self.actionClearRecents.triggered.connect(self.clear_recents_clicked)

    def remove_recent_project(self, file_path):
        """Remove a project from the Recent menu if OpenShot can't find it"""
        s = get_app().get_settings()
        recent_projects = s.get("recent_projects")
        if file_path in recent_projects:
            recent_projects.remove(file_path)
        s.set("recent_projects", recent_projects)
        s.save()

    def recent_project_clicked(self, file_path):
        """ Load a recent project when clicked """
        self.OpenProjectSignal.emit(file_path)

    def clear_recents_clicked(self):
        """Clear all recent projects"""
        s = get_app().get_settings()
        s.set("recent_projects", [])

        # Reload recent project list
        self.load_recent_menu()

    def openSecondDialog(self):
        mydialog = QDialog(self)
        label = QLabel('self')
        self.setCentralWidget(label)
        #mydialog.setModal(True)
        #mydialog.exec()
 
        mydialog.show()
 

    def setup_toolbars(self):
        _ = get_app()._tr  # Get translation function

        # Start undo and redo actions disabled
        self.actionUndo.setEnabled(False)
        self.actionRedo.setEnabled(False)

        # Add files toolbar
        self.filesToolbar = QToolBar("Files Toolbar")
        self.filesActionGroup = QActionGroup(self)
        self.filesActionGroup.setExclusive(True)
        self.filesActionGroup.addAction(self.actionFilesShowAll)
        self.filesActionGroup.addAction(self.actionFilesShowVideo)
        self.filesActionGroup.addAction(self.actionFilesShowAudio)
        self.filesActionGroup.addAction(self.actionFilesShowImage)
        self.actionFilesShowAll.setChecked(True)
        self.filesToolbar.addAction(self.actionFilesShowAll)
        self.filesToolbar.addAction(self.actionFilesShowVideo)
        self.filesToolbar.addAction(self.actionFilesShowAudio)
        self.filesToolbar.addAction(self.actionFilesShowImage)
        self.filesFilter = QLineEdit()
        self.filesFilter.setObjectName("filesFilter")
        self.filesFilter.setPlaceholderText(_("Filter"))
        self.filesFilter.setClearButtonEnabled(True)
        self.filesToolbar.addWidget(self.filesFilter)
        self.tabFiles.layout().insertWidget(0, self.filesToolbar)

        # Add transitions toolbar
        self.transitionsToolbar = QToolBar("Transitions Toolbar")
        self.transitionsActionGroup = QActionGroup(self)
        self.transitionsActionGroup.setExclusive(True)
        self.transitionsActionGroup.addAction(self.actionTransitionsShowAll)
        self.transitionsActionGroup.addAction(self.actionTransitionsShowCommon)
        self.actionTransitionsShowAll.setChecked(True)
        self.transitionsToolbar.addAction(self.actionTransitionsShowAll)
        self.transitionsToolbar.addAction(self.actionTransitionsShowCommon)
        self.transitionsFilter = QLineEdit()
        self.transitionsFilter.setObjectName("transitionsFilter")
        self.transitionsFilter.setPlaceholderText(_("Filter"))
        self.transitionsFilter.setClearButtonEnabled(True)
        self.transitionsToolbar.addWidget(self.transitionsFilter)
        self.tabTransitions.layout().addWidget(self.transitionsToolbar)

        # Add effects toolbar
        self.effectsToolbar = QToolBar("Effects Toolbar")
        self.effectsFilter = QLineEdit()
        self.effectsFilter.setObjectName("effectsFilter")
        self.effectsFilter.setPlaceholderText(_("Filter"))
        self.effectsFilter.setClearButtonEnabled(True)
        self.effectsToolbar.addWidget(self.effectsFilter)
        self.tabEffects.layout().addWidget(self.effectsToolbar)

        # Add emojis toolbar
        self.emojisToolbar = QToolBar("Emojis Toolbar")
        self.emojiFilterGroup = QComboBox()
        self.emojisFilter = QLineEdit()
        self.emojisFilter.setObjectName("emojisFilter")
        self.emojisFilter.setPlaceholderText(_("Filter"))
        self.emojisFilter.setClearButtonEnabled(True)
        self.emojisToolbar.addWidget(self.emojiFilterGroup)
        self.emojisToolbar.addWidget(self.emojisFilter)
        self.tabEmojis.layout().addWidget(self.emojisToolbar)

        # Add Video Preview toolbar
        self.videoToolbar = QToolBar("Video Toolbar")

        # Add fixed spacer(s) (one for each "Other control" to keep playback controls centered)
        ospacer1 = QWidget(self)
        ospacer1.setMinimumSize(32, 1)  # actionSaveFrame
        self.videoToolbar.addWidget(ospacer1)

        # Add left spacer
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoToolbar.addWidget(spacer)

        # Playback controls (centered)
        self.videoToolbar.addAction(self.actionJumpStart)
        self.videoToolbar.addAction(self.actionRewind)
        self.videoToolbar.addAction(self.actionPlay)
        self.videoToolbar.addAction(self.actionFastForward)
        self.videoToolbar.addAction(self.actionJumpEnd)
        self.actionPlay.setCheckable(True)

        # Add right spacer
        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoToolbar.addWidget(spacer)

        # Other controls (right-aligned)
        self.videoToolbar.addAction(self.actionSaveFrame)

        # self.Btn = QPushButton("Open Dialog Box", self)
        # self.Btn.setFont(QtGui.QFont("Sanserif", 15))
        # self.Btn.clicked.connect(self.openSecondDialog)
        # self.videoToolbar.addWidget(self.Btn)

        self.tabVideo.layout().addWidget(self.videoToolbar)

        # Add Timeline toolbar
        self.timelineToolbar = QToolBar("Timeline Toolbar", self)

        self.timelineToolbar.addAction(self.actionAddTrack)
        self.timelineToolbar.addSeparator()

        # rest of options
        self.timelineToolbar.addAction(self.actionSnappingTool)
        self.timelineToolbar.addAction(self.actionRazorTool)
        self.timelineToolbar.addSeparator()
        self.timelineToolbar.addAction(self.actionRemoveAllMarkers)
        self.timelineToolbar.addAction(self.actionLipSync)
        self.timelineToolbar.addAction(self.actionAddMarker_ls)
        self.timelineToolbar.addAction(self.actionAddMarker)
        self.timelineToolbar.addAction(self.actionPreviousMarker)
        self.timelineToolbar.addAction(self.actionNextMarker)
        self.timelineToolbar.addAction(self.actionCenterOnPlayhead)
        self.timelineToolbar.addAction(self.actionSyncNet)
        self.timelineToolbar.addAction(self.actionAddMarker_sn)
        self.timelineToolbar.addSeparator()
        self.timelineToolbar.addAction(self.actionFiller)
        self.timelineToolbar.addAction(self.actionSlideTranslate)
        self.timelineToolbar.addAction(self.actionFOMM)
        self.timelineToolbar.addAction(self.actionMakeItTalk)
        self.timelineToolbar.addAction(self.actionSpeechToText)

        languages={'bn': 'Bengali','en':"English",'hi': 'Hindi','ml': 'Malayalam','mr': 'Marathi','ta': 'Tamil','te': 'Telugu'}
        self.translateMenu = QMenu()
        self.translateBtn = QPushButton("Translate", self)
        self.translateBtn.setMenu(self.translateMenu)
        for i in languages:
            self.translateMenu.addAction(languages[i])
        
        self.timelineToolbar.addWidget(self.translateBtn)
        self.translateMenu.triggered[QAction].connect(self.actiontextTranslate_trigger)
       






        # Add Video Preview toolbar
        self.captionToolbar = QToolBar(_("Caption Toolbar"))

        # Add Caption text editor widget
        self.captionTextEdit = QTextEdit()
        self.captionTextEdit.setReadOnly(True)

        # Playback controls (centered)
        self.captionToolbar.addAction(self.actionInsertTimestamp)
        self.tabCaptions.layout().addWidget(self.captionToolbar)
        self.tabCaptions.layout().addWidget(self.captionTextEdit)

        # Hook up caption editor signal
        self.captionTextEdit.textChanged.connect(self.captionTextEdit_TextChanged)
        self.caption_save_timer = QTimer(self)
        self.caption_save_timer.setInterval(100)
        self.caption_save_timer.setSingleShot(True)
        self.caption_save_timer.timeout.connect(self.caption_editor_save)
        self.CaptionTextLoaded.connect(self.caption_editor_load)
        self.caption_model_row = None

        ### CHANGED ###

        # TTS and Text edit option
        self.textToolbar = QToolBar(_("TTS and Text Toolbar"))
        self.textToolbar.addAction(self.actionSaveText)
        self.textToolbar.addWidget(spacer)
        self.textToolbar.addAction(self.actionImportTextFiles)
        self.textToolbar.addWidget(spacer)
        self.textToolbar.addAction(self.actionClearText)

        self.textTextEdit = QTextEdit()
        self.textMenu = QMenu()
        self.textBtn = QPushButton("Select TTS", self)
        self.textBtn.setMenu(self.textMenu)
        self.textMenu.addAction(self.actionGlow)
        self.textMenu.addAction(self.actionGoogleTSS)
        self.textMenu.addAction(self.actionRTVC)
        self.textToolbar.addWidget(self.textBtn)
        self.textTextEdit.setReadOnly(False)
        self.tabText.layout().addWidget(self.textToolbar)
        self.tabText.layout().addWidget(self.textTextEdit)        

        
        self.slidetextToolbar = QToolBar(_("Slide Text Toolbar"))
        self.slidetextToolbar.addAction(self.actionSaveTextSlide)
        self.slidetextToolbar.addWidget(spacer)
        self.slidetextToolbar.addAction(self.actionClearTextSlide)

        self.slideTextEdit = QTextEdit()
        self.slideTextEdit.setReadOnly(False)
        self.tabSlideText.layout().addWidget(self.slidetextToolbar)
        self.tabSlideText.layout().addWidget(self.slideTextEdit)  
        ### CHANGED ###

        # Get project's initial zoom value
        initial_scale = get_app().project.get("scale") or 15.0

        # Setup Zoom Slider widget
        from windows.views.zoom_slider import ZoomSlider
        self.sliderZoomWidget = ZoomSlider(self)
        self.sliderZoomWidget.setMinimumSize(200, 20)
        self.sliderZoomWidget.setZoomFactor(initial_scale)

        # add zoom widgets
        self.timelineToolbar.addWidget(self.sliderZoomWidget)

        # Add timeline toolbar to web frame
        self.frameWeb.addWidget(self.timelineToolbar)

    def clearSelections(self):
        """Clear all selection containers"""
        self.selected_clips = []
        self.selected_transitions = []
        self.selected_markers = []
        self.selected_tracks = []
        self.selected_effects = []

        # Clear selection in properties view
        if self.propertyTableView:
            self.propertyTableView.loadProperties.emit("", "")

    def foundCurrentVersion(self, version):
        """Handle the callback for detecting the current version on openshot.org"""
        log.info('foundCurrentVersion: Found the latest version: %s' % version)
        _ = get_app()._tr

        # Compare versions (alphabetical compare of version strings should work fine)
        if info.VERSION < version:
            # Add spacer and 'New Version Available' toolbar button (default hidden)
            spacer = QWidget(self)
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.toolBar.addWidget(spacer)

            # Update text for QAction
            self.actionUpdate.setVisible(True)
            self.actionUpdate.setText(_("Update Available"))
            self.actionUpdate.setToolTip(_("Update Available: <b>%s</b>") % version)

            # Add update available button (with icon and text)
            updateButton = QToolButton()
            updateButton.setDefaultAction(self.actionUpdate)
            updateButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.toolBar.addWidget(updateButton)

    def moveEvent(self, event):
        """ Move tutorial dialogs also (if any)"""
        QMainWindow.moveEvent(self, event)
        if self.tutorial_manager:
            self.tutorial_manager.re_position_dialog()

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        if self.tutorial_manager:
            self.tutorial_manager.re_position_dialog()

    def showEvent(self, event):
        """ Have any child windows follow main-window state """
        QMainWindow.showEvent(self, event)
        for child in self.getDocks():
            if child.isFloating() and child.isEnabled():
                child.raise_()
                child.show()

    def hideEvent(self, event):
        """ Have any child windows hide with main window """
        QMainWindow.hideEvent(self, event)
        for child in self.getDocks():
            if child.isFloating() and child.isVisible():
                child.hide()

    def show_property_timeout(self):
        """Callback for show property timer"""

        # Emit load properties signal
        self.propertyTableView.loadProperties.emit(
            self.show_property_id,
            self.show_property_type)

    def InitKeyboardShortcuts(self):
        """Initialize all keyboard shortcuts from the settings file"""

        # Translate object
        _ = get_app()._tr

        # Update all action-based shortcuts (from settings file)
        for shortcut in self.getAllKeyboardShortcuts():
            for action in self.findChildren(QAction):
                if shortcut.get('setting') == action.objectName():
                    action.setShortcut(QKeySequence(_(shortcut.get('value'))))

    def InitCacheSettings(self):
        """Set the correct cache settings for the timeline"""
        # Load user settings
        s = get_app().get_settings()
        log.info("InitCacheSettings")
        log.info("cache-mode: %s" % s.get("cache-mode"))
        log.info("cache-limit-mb: %s" % s.get("cache-limit-mb"))

        # Get MB limit of cache (and convert to bytes)
        cache_limit = s.get("cache-limit-mb") * 1024 * 1024  # Convert MB to Bytes

        # Clear old cache
        new_cache_object = None
        if s.get("cache-mode") == "CacheMemory":
            # Create CacheMemory object, and set on timeline
            log.info("Creating CacheMemory object with %s byte limit" % cache_limit)
            new_cache_object = openshot.CacheMemory(cache_limit)
            self.timeline_sync.timeline.SetCache(new_cache_object)

        elif s.get("cache-mode") == "CacheDisk":
            # Create CacheDisk object, and set on timeline
            log.info("Creating CacheDisk object with %s byte limit at %s" % (
                cache_limit, info.PREVIEW_CACHE_PATH))
            image_format = s.get("cache-image-format")
            image_quality = s.get("cache-quality")
            image_scale = s.get("cache-scale")
            new_cache_object = openshot.CacheDisk(
                info.PREVIEW_CACHE_PATH,
                image_format,
                image_quality,
                image_scale,
                cache_limit,
                )
            self.timeline_sync.timeline.SetCache(new_cache_object)

        # Clear old cache before it goes out of scope
        if self.cache_object:
            self.cache_object.Clear()
        # Update cache reference, so it doesn't go out of scope
        self.cache_object = new_cache_object

    def initModels(self):
        """Set up model/view classes for MainWindow"""
        s = get_app().get_settings()

        # Setup files tree and list view (both share a model)
        self.files_model = FilesModel()
        self.filesTreeView = FilesTreeView(self.files_model)
        self.filesListView = FilesListView(self.files_model)
        self.files_model.update_model()
        self.tabFiles.layout().insertWidget(-1, self.filesTreeView)
        self.tabFiles.layout().insertWidget(-1, self.filesListView)
        if s.get("file_view") == "details":
            self.filesView = self.filesTreeView
            self.filesListView.hide()
        else:
            self.filesView = self.filesListView
            self.filesTreeView.hide()
        # Show our currently-enabled project files view
        self.filesView.show()
        self.filesView.setFocus()

        # Setup transitions tree and list views
        self.transition_model = TransitionsModel()
        self.transitionsTreeView = TransitionsTreeView(self.transition_model)
        self.transitionsListView = TransitionsListView(self.transition_model)
        self.transition_model.update_model()
        self.tabTransitions.layout().insertWidget(-1, self.transitionsTreeView)
        self.tabTransitions.layout().insertWidget(-1, self.transitionsListView)
        if s.get("transitions_view") == "details":
            self.transitionsView = self.transitionsTreeView
            self.transitionsListView.hide()
        else:
            self.transitionsView = self.transitionsListView
            self.transitionsTreeView.hide()
        # Show our currently-enabled transitions view
        self.transitionsView.show()
        self.transitionsView.setFocus()

        # Setup effects tree
        self.effects_model = EffectsModel()
        self.effectsTreeView = EffectsTreeView(self.effects_model)
        self.effectsListView = EffectsListView(self.effects_model)
        self.effects_model.update_model()
        self.tabEffects.layout().insertWidget(-1, self.effectsTreeView)
        self.tabEffects.layout().insertWidget(-1, self.effectsListView)
        if s.get("effects_view") == "details":
            self.effectsView = self.effectsTreeView
            self.effectsListView.hide()
        else:
            self.effectsView = self.effectsListView
            self.effectsTreeView.hide()
        # Show our currently-enabled effects view
        self.effectsView.show()
        self.effectsView.setFocus()

        # Setup emojis view
        self.emojis_model = EmojisModel()
        self.emojis_model.update_model()
        self.emojiListView = EmojisListView(self.emojis_model)
        self.tabEmojis.layout().addWidget(self.emojiListView)

    def __init__(self, *args, mode=None):

        ### CHANGED ###
        self.url = sys.argv[1]+"/"
        #self.url = 'bhaasha.iiit.ac.in/f2f_translation/'
        self.url_tts = self.url + 'tts1'
        self.url_tts2 = self.url + 'tts2'
        self.url_ls = self.url + 'lipsync'
        self.url_fomm = self.url + 'fomm'
        self.url_mit = self.url + 'mit'
        self.url_sp1 = self.url + 'slide_parser1'
        self.url_sp2 = self.url + 'slide_parser2'
        self.url_stt = self.url + 'speechtotext'
        self.url_trans = self.url + 'translate'
        self.filepath = None
        self.coordinates = [-1,-1,-1,-1]
        ### CHAGNED ###

        # Create main window base class
        super().__init__(*args)
        self.mode = mode    # None or unittest (None is normal usage)
        self.initialized = False

        # set window on app for reference during initialization of children
        app = get_app()
        app.window = self
        _ = app._tr

        # Load user settings for window
        s = app.get_settings()
        self.recent_menu = None

        # Track metrics
        track_metric_session()  # start session

        # Set unique install id (if blank)
        if not s.get("unique_install_id"):
            s.set("unique_install_id", str(uuid4()))

            # Track 1st launch metric
            track_metric_screen("initial-launch-screen")

        # Track main screen
        track_metric_screen("main-screen")

        # Create blank tutorial manager
        self.tutorial_manager = None

        # Load UI from designer
        ui_util.load_ui(self, self.ui_path)

        # Set all keyboard shortcuts from the settings file
        self.InitKeyboardShortcuts()

        # Init UI
        ui_util.init_ui(self)

        # Setup toolbars that aren't on main window, set initial state of items, etc
        self.setup_toolbars()

        # Add window as watcher to receive undo/redo status updates
        app.updates.add_watcher(self)

        # Get current version of OpenShot via HTTP
        self.FoundVersionSignal.connect(self.foundCurrentVersion)
        get_current_Version()

        # Connect signals
        if self.mode != "unittest":
            self.RecoverBackup.connect(self.recover_backup)

        # Initialize and start the thumbnail HTTP server
        self.http_server_thread = httpThumbnailServerThread()
        self.http_server_thread.start()

        # Create the timeline sync object (used for previewing timeline)
        self.timeline_sync = TimelineSync(self)

        # Setup timeline
        self.timeline = TimelineWebView(self)
        self.frameWeb.layout().addWidget(self.timeline)

        # Configure the side docks to full-height
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)

        self.initModels()

        # Add Docks submenu to View menu
        self.addViewDocksMenu()

        # Set up status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Process events before continuing
        # TODO: Figure out why this is needed for a backup recovery to correctly show up on the timeline
        app.processEvents()

        # Setup properties table
        self.txtPropertyFilter.setPlaceholderText(_("Filter"))
        self.propertyTableView = PropertiesTableView(self)
        self.selectionLabel = SelectionLabel(self)
        self.dockPropertiesContent.layout().addWidget(self.selectionLabel, 0, 1)
        self.dockPropertiesContent.layout().addWidget(self.propertyTableView, 2, 1)

        # Init selection containers
        self.clearSelections()

        # Show Property timer
        # Timer to use a delay before showing properties (to prevent a mass selection from trying
        # to update the property model hundreds of times)
        self.show_property_id = None
        self.show_property_type = None
        self.show_property_timer = QTimer(self)
        self.show_property_timer.setInterval(100)
        self.show_property_timer.setSingleShot(True)
        self.show_property_timer.timeout.connect(self.show_property_timeout)

        # Selection timer
        # Timer to use a delay before emitting selection signal (to prevent a mass selection from trying
        # to update the zoom slider widget hundreds of times)
        self.selection_timer = QTimer(self)
        self.selection_timer.setInterval(100)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.timeout.connect(self.emit_selection_signal)

        # Setup video preview QWidget
        self.videoPreview = VideoWidget()
        self.tabVideo.layout().insertWidget(0, self.videoPreview)

        # Load window state and geometry
        self.load_settings()

        # Setup Cache settings
        self.cache_object = None
        self.InitCacheSettings()

        # Start the preview thread
        self.preview_parent = PreviewParent()
        self.preview_parent.Init(self, self.timeline_sync.timeline, self.videoPreview)
        self.preview_thread = self.preview_parent.worker
        self.sliderZoomWidget.connect_playback()

        # Set pause callback
        self.PauseSignal.connect(self.handlePausedVideo)

        # QTimer for Autosave
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.setInterval(int(s.get("autosave-interval") * 1000 * 60))
        self.auto_save_timer.timeout.connect(self.auto_save_project)
        if s.get("enable-auto-save"):
            self.auto_save_timer.start()

        # Set encoding method
        if s.get("hw-decoder"):
            openshot.Settings.Instance().HARDWARE_DECODER = int(str(s.get("hw-decoder")))
        else:
            openshot.Settings.Instance().HARDWARE_DECODER = 0

        # Set graphics card for decoding
        if s.get("graca_number_de"):
            if int(str(s.get("graca_number_de"))) != 0:
                openshot.Settings.Instance().HW_DE_DEVICE_SET = int(str(s.get("graca_number_de")))
            else:
                openshot.Settings.Instance().HW_DE_DEVICE_SET = 0
        else:
            openshot.Settings.Instance().HW_DE_DEVICE_SET = 0

        # Set graphics card for encoding
        if s.get("graca_number_en"):
            if int(str(s.get("graca_number_en"))) != 0:
                openshot.Settings.Instance().HW_EN_DEVICE_SET = int(str(s.get("graca_number_en")))
            else:
                openshot.Settings.Instance().HW_EN_DEVICE_SET = 0
        else:
            openshot.Settings.Instance().HW_EN_DEVICE_SET = 0

        # Set audio playback settings
        if s.get("playback-audio-device"):
            openshot.Settings.Instance().PLAYBACK_AUDIO_DEVICE_NAME = str(s.get("playback-audio-device"))
        else:
            openshot.Settings.Instance().PLAYBACK_AUDIO_DEVICE_NAME = ""

        # Set scaling mode to lower quality scaling (for faster previews)
        openshot.Settings.Instance().HIGH_QUALITY_SCALING = False

        # Set use omp threads number environment variable
        if s.get("omp_threads_number"):
            openshot.Settings.Instance().OMP_THREADS = max(2, int(str(s.get("omp_threads_number"))))
        else:
            openshot.Settings.Instance().OMP_THREADS = 12

        # Set use ffmpeg threads number environment variable
        if s.get("ff_threads_number"):
            openshot.Settings.Instance().FF_THREADS = max(1, int(str(s.get("ff_threads_number"))))
        else:
            openshot.Settings.Instance().FF_THREADS = 8

        # Set use max width decode hw environment variable
        if s.get("decode_hw_max_width"):
            openshot.Settings.Instance().DE_LIMIT_WIDTH_MAX = int(str(s.get("decode_hw_max_width")))

        # Set use max height decode hw environment variable
        if s.get("decode_hw_max_height"):
            openshot.Settings.Instance().DE_LIMIT_HEIGHT_MAX = int(str(s.get("decode_hw_max_height")))

        # Create lock file
        self.create_lock_file()

        # Connect OpenProject Signal
        self.OpenProjectSignal.connect(self.open_project)

        # Connect Selection signals
        self.SelectionAdded.connect(self.addSelection)
        self.SelectionRemoved.connect(self.removeSelection)

        # Show window
        if self.mode != "unittest":
            self.show()
        else:
            log.info('Hiding UI for unittests')

        # Create tutorial manager
        self.tutorial_manager = TutorialManager(self)

        # Save settings
        s.save()

        # Refresh frame
        QTimer.singleShot(100, self.refreshFrameSignal.emit)

        # Main window is initialized
        self.initialized = True
