// Main JavaScript for FramePack
// Initializes modules and sets up global app functionality

import { 
    initElements, 
    enforceHorizontalLayout 
} from './common.js';

import { initEditor } from './editor.js';
import { initJobQueue } from './job_queue.js';
import { initOutputs } from './outputs.js';
import { initBatch } from './batch.js';

// Theme-related functions
function initTheme() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const htmlElement = document.documentElement;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    
    // Apply saved theme or detect system preference
    if (savedTheme) {
        htmlElement.setAttribute('data-bs-theme', savedTheme);
        darkModeToggle.checked = savedTheme === 'dark';
    } else {
        // Use system preference as fallback
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
            htmlElement.setAttribute('data-bs-theme', 'dark');
            darkModeToggle.checked = true;
        }
    }
    
    // Update icon based on current theme
    updateThemeIcon(darkModeToggle.checked);
}

function toggleDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const htmlElement = document.documentElement;
    
    if (darkModeToggle.checked) {
        htmlElement.setAttribute('data-bs-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    } else {
        htmlElement.setAttribute('data-bs-theme', 'light');
        localStorage.setItem('theme', 'light');
    }
    
    // Update icon based on current theme
    updateThemeIcon(darkModeToggle.checked);
}

function updateThemeIcon(isDark) {
    const icon = document.querySelector('label[for="darkModeToggle"] i');
    if (icon) {
        if (isDark) {
            icon.className = 'bi bi-moon-stars-fill';
        } else {
            icon.className = 'bi bi-brightness-high';
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing FramePack application');
    
    // Initialize UI elements
    initElements();
    
    initEditor();
    
    // Then initialize other modules
    initBatch();
    initJobQueue();
    initOutputs();
    
    // Initialize theme
    initTheme();
    
    // Theme toggle event listener
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('change', toggleDarkMode);
    }
    
    // Tab change events for layout updates
    const tabs = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', () => {
            // Small delay to ensure DOM is updated
            setTimeout(enforceHorizontalLayout, 50);
        });
    });
    
    // Apply horizontal layout on window resize
    window.addEventListener('resize', enforceHorizontalLayout);
    
    // Ensure toast container exists and is showing
    if (window.createToastContainer) {
        const container = window.createToastContainer();
    }
});

