// Text rotation animation
document.addEventListener('DOMContentLoaded', function() {
    const rotatingText = document.getElementById('rotatingText');
    const rotatingText2 = document.getElementById('rotatingText2');
    
    const words1 = ['bolt.new', 'claude code', 'any website', 'any app'];
    const words2 = ['talk', 'listen', 'text', 'slack'];
    
    let currentIndex1 = 0;
    let currentIndex2 = 0;

    function animateText1() {
        // Fade out
        rotatingText.style.opacity = '0';
        rotatingText.style.transform = 'translateY(-15px)';
        
        setTimeout(() => {
            // Change text
            currentIndex1 = (currentIndex1 + 1) % words1.length;
            rotatingText.textContent = words1[currentIndex1];
            
            // Add letter spacing for shorter words
            if (words1[currentIndex1] === 'bolt.new' || words1[currentIndex1] === 'any app') {
                rotatingText.style.letterSpacing = '0.01em';
            } else {
                rotatingText.style.letterSpacing = '-0.1em';
            }
            
            // Fade in
            rotatingText.style.opacity = '1';
            rotatingText.style.transform = 'translateY(0)';
        }, 300);
    }

    function animateText2() {
        // Fade out
        rotatingText2.style.opacity = '0';
        rotatingText2.style.transform = 'translateY(-15px)';
        
        setTimeout(() => {
            // Change text
            currentIndex2 = (currentIndex2 + 1) % words2.length;
            rotatingText2.textContent = words2[currentIndex2];
            
            // Add letter spacing for shorter words
            if (words2[currentIndex2] === 'listen' || words2[currentIndex2] === 'slack') {
                rotatingText2.style.letterSpacing = '-0.08em';
            } else {
                rotatingText2.style.letterSpacing = '0.01em';
            }
            
            // Fade in
            rotatingText2.style.opacity = '1';
            rotatingText2.style.transform = 'translateY(0)';
        }, 300);
    }

    // Set initial letter spacing - bolt.new starts normal, talk gets spacing
    rotatingText.style.letterSpacing = '0.01em';
    rotatingText2.style.letterSpacing = '0.01em';

    // Start first animation after 1 second, then repeat every 3 seconds
    setTimeout(() => {
        setInterval(animateText1, 3000);
    }, 500);

    // Start second animation after 2.5 seconds (1.5 seconds after first), then repeat every 3 seconds
    setTimeout(() => {
        setInterval(animateText2, 3000);
    }, 2000);

    // Morphing signup functionality
    const morphingBtn = document.getElementById('morphingBtn');
    const morphingForm = document.getElementById('morphingForm');
    const morphingEmailInput = document.getElementById('morphingEmailInput');

    morphingBtn.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Hide content instantly first
        morphingBtn.classList.add('hide-content');
        
        // Then start morphing animation
        setTimeout(() => {
            morphingBtn.classList.add('morphing');
            
            // Show form after button starts fading
            setTimeout(() => {
                morphingForm.classList.add('active');
                // Focus the email input after animation
                setTimeout(() => {
                    morphingEmailInput.focus();
                }, 0);
            }, 0);
        }, 0); // Very short delay to ensure content is hidden first
    });

    // Handle form submission
    morphingForm.addEventListener('submit', function(e) {
        const submitBtn = this.querySelector('.morphing-submit-btn');
        const originalHTML = submitBtn.innerHTML;
        
        // Show loading state
        submitBtn.innerHTML = `
            <svg class="btn-icon animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12a9 9 0 11-6.219-8.56"/>
            </svg>
        `;
        submitBtn.disabled = true;
        
        // Reset after form submission (Formspree will handle the redirect)
        setTimeout(() => {
            submitBtn.innerHTML = originalHTML;
            submitBtn.disabled = false;
        }, 2000);
    });

    // Click outside to close form
    document.addEventListener('click', function(e) {
        const container = document.querySelector('.morphing-signup-container');
        if (!container.contains(e.target) && morphingForm.classList.contains('active')) {
            morphingForm.classList.remove('active');
            setTimeout(() => {
                morphingBtn.classList.remove('morphing');
                morphingBtn.classList.remove('hide-content');
            }, 200);
        }
    });

    // Escape key to close form
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && morphingForm.classList.contains('active')) {
            morphingForm.classList.remove('active');
            setTimeout(() => {
                morphingBtn.classList.remove('morphing');
                morphingBtn.classList.remove('hide-content');
            }, 200);
        }
    });

    // Smooth scrolling for any internal links (if added later)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add some entrance animation to hero content
    const heroContent = document.querySelector('.hero-content');
    if (heroContent) {
        heroContent.style.opacity = '0';
        heroContent.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            heroContent.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
            heroContent.style.opacity = '1';
            heroContent.style.transform = 'translateY(0)';
        }, 100);
    }

    // Add scroll-triggered animations for feature cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe feature cards and pricing cards
    document.querySelectorAll('.feature-card, .pricing-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    // Enhanced scroll effects
    let ticking = false;
    
    function updateScrollEffects() {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        // Parallax effect for hero background
        const hero = document.querySelector('.hero');
        if (hero) {
            hero.style.transform = `translateY(${rate}px)`;
        }
        
        // Navigation background opacity
        const nav = document.querySelector('.nav');
        if (nav) {
            if (scrolled > 50) {
                nav.classList.add('scrolled');
            } else {
                nav.classList.remove('scrolled');
            }
        }
        
        ticking = false;
    }
    
    function requestTick() {
        if (!ticking) {
            requestAnimationFrame(updateScrollEffects);
            ticking = true;
        }
    }
    
    window.addEventListener('scroll', requestTick);

    // Add scroll progress indicator
    const scrollProgress = document.createElement('div');
    scrollProgress.className = 'scroll-progress';
    document.body.appendChild(scrollProgress);

    window.addEventListener('scroll', () => {
        const scrollTop = document.documentElement.scrollTop;
        const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrollPercent = (scrollTop / scrollHeight) * 100;
        scrollProgress.style.width = scrollPercent + '%';
    });

    // Add staggered animation to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });

    // Enhanced button hover effects
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px) scale(1.02)';
        });
        
        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Add ripple effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
});