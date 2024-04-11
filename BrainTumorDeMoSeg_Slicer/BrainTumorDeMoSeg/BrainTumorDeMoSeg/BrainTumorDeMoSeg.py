import logging
import os
import re

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

PACKAGE = {
    'torch': '1.12.1',
    'numpy': '1.24.4',
    'pandas': '2.0.3',
    'pillow': '10.2.0',
    # 'requests': '2.31.0',
    'scikit-image': '0.21.0',
    'scikit-learn': '1.3.2',
    'SimpleITK': '2.3.0'
}

#
# BrainTumorDeMoSeg
#


class BrainTumorDeMoSeg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "BrainTumorDeMoSeg"  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Segmentation"] # [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Kaixiang Yang"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = """
This is an automatic segmenting tools for incomplete brain tumors segmentation. 
Importantly, 4-modality MRI images should be skull-stripped and co-registered.
The model has been trained on BraTS2020.
See more information in the <a href="https://github.com/kk42yy/DeMoSeg">DeMoSeg extension documentation</a>
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Kaixiang Yang, Wenqi Shan, Qiang Li, Zhiwei Wang and etc.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.empty_function)

    def empty_function(self):
        pass


#
# BrainTumorDeMoSegWidget
#


class BrainTumorDeMoSegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/BrainTumorDeMoSeg.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = BrainTumorDeMoSegLogic()
        self.logic.logCallback = self.addLog

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.T1VolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.T1ceVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.T2VolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.FlairVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        self.ui.TTACheckBox.connect('toggled(bool)', self.updateParameterNodeFromGUI)
        self.ui.cpuCheckBox.connect('toggled(bool)', self.updateParameterNodeFromGUI)
        self.ui.useStandardSegmentNamesCheckBox.connect('toggled(bool)', self.updateParameterNodeFromGUI)


        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.ui.segmentationShow3DButton.setSegmentationNode)

        # Buttons
        self.ui.packageInfoUpdateButton.connect('clicked(bool)', self.onPackageInfoUpdate)
        self.ui.packageUpgradeButton.connect('clicked(bool)', self.onPackageUpgrade)
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
          self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not (
            self._parameterNode.GetNodeReference("T1Volume") or \
            self._parameterNode.GetNodeReference("T1ceVolume") or \
            self._parameterNode.GetNodeReference("T2Volume") or \
            self._parameterNode.GetNodeReference("FlairVolume"),
            ):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("T1Volume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.T1VolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("T1Volume"))
        self.ui.T1ceVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("T1ceVolume"))
        self.ui.T2VolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("T2Volume"))
        self.ui.FlairVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("FlairVolume"))

        self.ui.TTACheckBox.checked = self._parameterNode.GetParameter("TTA") == "true"
        self.ui.cpuCheckBox.checked = self._parameterNode.GetParameter("CPU") == "true"
        self.ui.useStandardSegmentNamesCheckBox.checked = self._parameterNode.GetParameter("UseStandardSegmentNames") == "true"
        self.ui.outputSegmentationSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputSegmentation"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("T1Volume"):
            inputVolume = self._parameterNode.GetNodeReference("T1Volume")
        elif self._parameterNode.GetNodeReference("T1ceVolume"):
            inputVolume = self._parameterNode.GetNodeReference("T1ceVolume")
        elif self._parameterNode.GetNodeReference("T2Volume"):
            inputVolume = self._parameterNode.GetNodeReference("T2Volume")
        elif self._parameterNode.GetNodeReference("FlairVolume"):
            inputVolume = self._parameterNode.GetNodeReference("FlairVolume")
        else:
            inputVolume = self._parameterNode.GetNodeReference("T1Volume") # None
        
        if inputVolume:
            self.ui.applyButton.toolTip = "Start segmentation"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input volume"
            self.ui.applyButton.enabled = False

        if inputVolume:
            tmpname = inputVolume.GetName()
            try:
                for mod in ['_t1', 't1', '_T1', 'T1']:
                    tmpname = tmpname.replace(mod, '')
                for mod in ['_t1ce', 't1ce', '_T1ce', 'T1ce']:
                    tmpname = tmpname.replace(mod, '')
                for mod in ['_t2', 't2', '_T2', 'T2']:
                    tmpname = tmpname.replace(mod, '')
                for mod in ['_flair', 'flair', '_FLAIR', 'FLAIR']:
                    tmpname = tmpname.replace(mod, '')
            except:
                pass

            self.ui.outputSegmentationSelector.baseName = tmpname + "_segmentation"

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("T1Volume", self.ui.T1VolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("T1ceVolume", self.ui.T1ceVolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("T2Volume", self.ui.T2VolumeSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("FlairVolume", self.ui.FlairVolumeSelector.currentNodeID)

        self._parameterNode.SetParameter("TTA", "true" if self.ui.TTACheckBox.checked else "false")
        self._parameterNode.SetParameter("CPU", "true" if self.ui.cpuCheckBox.checked else "false")
        self._parameterNode.SetParameter("UseStandardSegmentNames", "true" if self.ui.useStandardSegmentNamesCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputSegmentation", self.ui.outputSegmentationSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def addLog(self, text):
        """Append text to log window
        """
        self.ui.statusLabel.appendPlainText(text)
        slicer.app.processEvents()  # force update

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        self.ui.statusLabel.plainText = ''

        import qt
        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            self.logic.setupPythonRequirements()
            slicer.app.restoreOverrideCursor()
        except Exception as e:
            slicer.app.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            self.ui.statusLabel.appendPlainText("\nApplication restart required.")
            if slicer.util.confirmOkCancelDisplay(
                "Application is required to complete installation of required Python packages.\nPress OK to restart.",
                "Confirm application restart"
                ):
                slicer.util.restart()
            else:
                return

        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Create new segmentation node, if not selected yet
            if not self.ui.outputSegmentationSelector.currentNode():
                self.ui.outputSegmentationSelector.addNode()

            self.logic.useStandardSegmentNames = self.ui.useStandardSegmentNamesCheckBox.checked

            # Compute output
            self.logic.process(
                self.ui.T1VolumeSelector.currentNode(), 
                self.ui.T1ceVolumeSelector.currentNode(), 
                self.ui.T2VolumeSelector.currentNode(), 
                self.ui.FlairVolumeSelector.currentNode(), 
                self.ui.outputSegmentationSelector.currentNode(),
                self.ui.TTACheckBox.checked, self.ui.cpuCheckBox.checked)

        self.ui.statusLabel.appendPlainText("\nProcessing finished.")

    def onPackageInfoUpdate(self):
        self.ui.packageInfoTextBrowser.plainText = ''
        with slicer.util.tryWithErrorDisplay("Failed to get BrainTumorDeMoSeg package version information", waitCursor=True):
            self.ui.packageInfoTextBrowser.plainText = self.logic.installedBrainTumorDeMoSegPythonPackageInfo().rstrip()

    def onPackageUpgrade(self):
        import ctk
        import qt
        mbox = ctk.ctkMessageBox(slicer.util.mainWindow())
        mbox.text = "Reinstall all package?"
        mbox.addButton("Yes", qt.QMessageBox.AcceptRole)
        mbox.addButton("No", qt.QMessageBox.RejectRole)
        mbox.deleteLater()
        Upgrade = (mbox.exec_() == qt.QMessageBox.AcceptRole)
        
        if not Upgrade:
            return
        
        with slicer.util.tryWithErrorDisplay("Failed to upgrade BrainTumorDeMoSeg", waitCursor=True):
            self.logic.setupPythonRequirements(upgrade=True)
        self.onPackageInfoUpdate()
        if not slicer.util.confirmOkCancelDisplay(f"This BrainTumorDeMoSeg update requires a 3D Slicer restart.","Press OK to restart."):
            raise ValueError('Restart was cancelled.')
        else:
            slicer.util.restart()

#
# BrainTumorDeMoSegLogic
#


class BrainTumorDeMoSegLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        # Custom applications can set custom location for weights.
        # For example, it could be set to `sysconfig.get_path('scripts')` to have an independent copy of
        # the weights for each Slicer installation. However, setting such custom path would result in extra downloads and
        # storage space usage if there were multiple Slicer installations on the same computer.
        self.BrainTumorDeMoSegWeightsPath = None

        self.logCallback = None
        self.clearOutputFolder = True
        self.useStandardSegmentNames = True
        self.pullMaster = False

    def log(self, text):
        logging.info(text)
        if self.logCallback:
            self.logCallback(text)


    def installedBrainTumorDeMoSegPythonPackageInfo(self):

        versionInfo = 'BrainTumorDeMoSeg package info\n'
        for pk in PACKAGE:
            versionInfo += f"{pk}: {self.packagePythonPackageVersion(pk)}\n"

        versionInfo += '\n'
        versionInfo += 'More info are in https://github.com/kk42yy/DeMoSeg'

        return versionInfo

    def packagePythonPackageVersion(self, package):
        """Utility function to get version of currently installed SimpleITK.
        Currently not used, but it can be useful for diagnostic purposes.
        """

        import shutil
        import subprocess
        versionInfo = subprocess.check_output([shutil.which('PythonSlicer'), "-m", "pip", "show", package]).decode()

        # versionInfo looks something like this:
        #
        #   Name: SimpleITK
        #   Version: 2.2.0rc2.dev368
        #   Summary: SimpleITK is a simplified interface to the Insight Toolkit (ITK) for image registration and segmentation
        #   ...
        #

        # Get version string (second half of the second line):
        version = versionInfo.split('\n')[1].split(' ')[1].strip()
        return version

    def setupPythonRequirements(self, upgrade=False):

        if upgrade:
            self.log("Starting upgrade package\n")
            for pk in PACKAGE:
                self.log(f"uninstalling {pk}")
                slicer.util.pip_uninstall(pk)

        self.pth_path = os.path.split(os.path.abspath(__file__))[0]+"/BraTS20_Model.pth"
        if os.path.isfile(self.pth_path):
            self.log(f"model has been saved in {self.pth_path}")
        else:
            import requests, zipfile
            self.log(f"model is missing, downloading now from \
                     https://github.com/kk42yy/DeMoSeg_Slicer/releases/download/v1.0.0/BraTS20_Model.zip")
            
            url = "https://github.com/kk42yy/DeMoSeg_Slicer/releases/download/v1.0.0/BraTS20_Model.zip"
            
            config_dir = os.path.split(os.path.abspath(__file__))[0]
            with open(config_dir+'/tmp_download_file.zip', 'wb') as f:

                with requests.get(url, stream=True) as r:
                    r.raise_for_status()

                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        f.write(chunk)
            
            with zipfile.ZipFile(config_dir+'/tmp_download_file.zip', 'r') as zip_f:
                zip_f.extractall(config_dir)
            os.remove(config_dir+'/tmp_download_file.zip')
            self.log("Download finished.")

        try:
            import torch, numpy, SimpleITK, skimage, scipy
            self.log("Package are prepared!")
            return
        except:
            pass
        
        self.log("Starting install package\n")
        for pk, ver in PACKAGE.items():
            self.log(f"installing {pk}=={ver}")
            slicer.util.pip_install(f"{pk}=={ver}")

        self.log('BrainTumorDeMoSeg installation completed successfully.')


    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("TTA"):
            parameterNode.SetParameter("TTA", "false")
        if not parameterNode.GetParameter("UseStandardSegmentNames"):
            parameterNode.SetParameter("UseStandardSegmentNames", "true")

    def process(self, t1, t1ce, t2, flair, outputSegmentation, TTA=False, cpu=True):

        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param t1, t1ce, t2, flair: volume to be thresholded
        :param outputVolume: thresholding result
        :param TTA: mirroring testing time augmentation
        :param cpu: whether force to use cpu
        """

        if not (t1 or t1ce or t2 or flair):
            raise ValueError("Input or output volume is invalid")
        
        import time
        startTime = time.time()
        self.log('Processing started')

        # Ensure which missing situations
        T1___, T1ce_, T2___, FLAIR = False, False, False, False
        if t1: T1___ = True
        if t1ce: T1ce_ = True
        if t2: T2___ = True
        if flair: FLAIR = True

        missing_modality_mapping = {
            (True , False, False, False): 0,
            (False, True , False, False): 1,
            (False, False, True , False): 2,
            (False, False, False, True ): 3,
            (True , True , False, False): 4,
            (True , False, True , False): 5,
            (True , False, False, True ): 6,
            (False, True , True , False): 7,
            (False, True , False, True ): 8,
            (False, False, True , True ): 9,
            (True , True , True , False): 10,
            (True , True , False, True ): 11,
            (True , False, True , True ): 12,
            (False, True , True , True ): 13,
            (True , True , True , True ): 14,
        }

        modality = missing_modality_mapping[(T1___, T1ce_, T2___, FLAIR)]
        self.log(f"\nReceive: T1={T1___}, T1ce={T1ce_}, T2={T2___}, FLAIR={FLAIR}")
        self.log(f"The missing modality is: {modality}\n")
        
        # Create new empty folder
        tempFolder = slicer.util.tempDirectory()
        inputFileBase = tempFolder+"/total-segmentator-input"
        outputSegmentationFile = tempFolder + "/segmentation.nii"

        # Recommend the user to forbid TTA if no GPU or not enough memory is available
        import torch

        cuda = torch.cuda if torch.backends.cuda.is_built() and torch.cuda.is_available() else None
        if cpu:
            cuda = False
            self.log(f"Force to use cpu!")

        if TTA and not cuda:

            import ctk
            import qt
            mbox = ctk.ctkMessageBox(slicer.util.mainWindow())
            mbox.text = "No GPU is detected. Switch to 'fast' mode to forbid TTA or compute mirror TTA in more few minutes?"
            mbox.addButton("Fast (~1 minutes)", qt.QMessageBox.AcceptRole)
            mbox.addButton("TTA (~10 minutes)", qt.QMessageBox.RejectRole)
            # Windows 10 peek feature in taskbar shows all hidden but not destroyed windows
            # (after creating and closing a messagebox, hovering over the mouse on Slicer icon, moving up the
            # mouse to the peek thumbnail would show it again).
            mbox.deleteLater()
            TTA = not (mbox.exec_() == qt.QMessageBox.AcceptRole)
            if not cpu:
                cpu = True
            
            self.log(f"TTA {TTA}; CPU {cpu}")

        if TTA and cuda and cuda.get_device_properties(cuda.current_device()).total_memory < 7e9:
            if slicer.util.confirmYesNoDisplay("You have less than 7 GB of GPU memory available. Forbid TTA to ensure segmentation can be completed successfully?"):
                TTA = False

        # Get BrainTumorDeMoSeg launcher command
        # BrainTumorDeMoSeg (.py file, without extension) is installed in Python Scripts folder
        import shutil

        # Write input volume to file
        # BrainTumorDeMoSeg requires NIFTI
        self.log(f"Writing input file to {tempFolder}")

        # write img if existing
        SaveAsZero = []
        for modidx, ifexist, volumenode in zip(
            [f"_000{i}.nii.gz" for i in range(4)], \
            [T1___, T1ce_, T2___, FLAIR], \
            [t1, t1ce, t2, flair]
        ):
            if ifexist:
                volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
                volumeStorageNode.SetFileName(inputFileBase+modidx)
                volumeStorageNode.UseCompressionOff()
                volumeStorageNode.WriteData(volumenode)
                volumeStorageNode.UnRegister(None)
                
                Reference_NIIGZ_Path = inputFileBase+modidx
                inputVolume = volumenode
            else:
                SaveAsZero.append(modidx)

        # write missing modality as ZERO matrix
        import SimpleITK as sitk
        import numpy as np
        self.log(f"writing missing modality as ZERO matrix: {SaveAsZero}")
        itk = sitk.ReadImage(Reference_NIIGZ_Path)
        arr = sitk.GetArrayFromImage(itk)
        newarr = np.zeros_like(arr, dtype=arr.dtype)
        newitk = sitk.GetImageFromArray(newarr)
        newitk.CopyInformation(itk)
        for modidx in SaveAsZero:
            sitk.WriteImage(newitk, inputFileBase+modidx)

        # Launch DeMoSeg
        self.log('Creating segmentations with BrainTumorDeMoSeg AI...')
        from DeMoSeg import DeMoSeg_Infer
        DeMoSeg_Infer(inputFileBase+'_0000.nii.gz', outputSegmentationFile, self.pth_path,\
                      TTA, modality, 'cpu' if cpu else 'cuda', self.log)
        self.readSegmentation(outputSegmentation, outputSegmentationFile)

        # Load result
        self.log('Importing segmentation results...')
        
        # Set source volume - required for DICOM Segmentation export
        outputSegmentation.SetNodeReferenceID(outputSegmentation.GetReferenceImageGeometryReferenceRole(), inputVolume.GetID())
        outputSegmentation.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        # Place segmentation node in the same place as the input volume
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        inputVolumeShItem = shNode.GetItemByDataNode(inputVolume)
        studyShItem = shNode.GetItemParent(inputVolumeShItem)
        segmentationShItem = shNode.GetItemByDataNode(outputSegmentation)
        shNode.SetItemParent(segmentationShItem, studyShItem)

        if self.clearOutputFolder:
            self.log("Cleaning up temporary folder...")
            if os.path.isdir(tempFolder):
                shutil.rmtree(tempFolder)
        else:
            self.log(f"Not cleaning up temporary folder: {tempFolder}")

        stopTime = time.time()
        self.log(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def readSegmentation(self, outputSegmentation, outputSegmentationFile):

        labelValueToSegmentName = {
            1: 'NCR',
            2: 'ED',
            3: 'ET'
        }
        maxLabelValue = max(labelValueToSegmentName.keys())

        # Get color node with random colors
        rgba = {
            1: [256-128, 256-174, 256-128, 0],
            2: [256-241, 256-214, 256-145, 0],
            3: [256-177, 256-122, 256-101, 0]
        }

        # Create color table for this segmentation task
        colorTableNode = slicer.vtkMRMLColorTableNode()
        colorTableNode.SetTypeToUser()
        colorTableNode.SetNumberOfColors(maxLabelValue+1)
        colorTableNode.SetName('BraTS')
        for labelValue in labelValueToSegmentName:
            # randomColorsNode.GetColor(labelValue,rgba)
            colorTableNode.SetColor(labelValue, *rgba[labelValue])
            colorTableNode.SetColorName(labelValue, labelValueToSegmentName[labelValue])
        slicer.mrmlScene.AddNode(colorTableNode)

        # Load the segmentation
        outputSegmentation.SetLabelmapConversionColorTableNodeID(colorTableNode.GetID())
        outputSegmentation.AddDefaultStorageNode()
        storageNode = outputSegmentation.GetStorageNode()
        storageNode.SetFileName(outputSegmentationFile)
        storageNode.ReadData(outputSegmentation)

        slicer.mrmlScene.RemoveNode(colorTableNode)


#
# BrainTumorDeMoSegTest
#


class BrainTumorDeMoSegTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_BrainTumorDeMoSeg1()

    def test_BrainTumorDeMoSeg1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        inputVolume = SampleData.downloadSample('MRBrainTumor1')
        self.delayDisplay('Loaded test data set')

        outputSegmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')

        # Test the module logic

        # Logic testing is disabled by default to not overload automatic build machines (pytorch is a huge package and computation
        # on CPU takes 5-10 minutes). Set testLogic to True to enable testing.
        testLogic = True

        if testLogic:
            logic = BrainTumorDeMoSegLogic()
            logic.logCallback = self._mylog

            self.delayDisplay('Set up required Python packages')
            logic.setupPythonRequirements()

            self.delayDisplay('Compute output')
            logic.process(None, None, None, inputVolume, outputSegmentation, TTA=False, cpu=True)

        else:
            logging.warning("test_BrainTumorDeMoSeg1 logic testing was skipped")

        self.delayDisplay('Test passed')

    def _mylog(self,text):
        print(text)
